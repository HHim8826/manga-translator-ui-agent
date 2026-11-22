from __future__ import annotations

import concurrent.futures
import copy
import math
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from services import get_render_parameter_service

from .document_state import ExportBase
from .image_utils import image_like_to_pil, image_like_to_rgb_array
from .inpaint_state import InpaintArtifact
from .region_geometry_state import RegionGeometryState, normalize_region_geometry_data

if TYPE_CHECKING:
    from .editor_controller import EditorController

_UNKNOWN = object()
"""The JSON stamp of a page read without stamps: nothing to compare with."""


def _close_images(*images: object) -> None:
    closed: set[int] = set()
    for image in images:
        identity = id(image)
        if identity in closed:
            continue
        closed.add(identity)
        close = getattr(image, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


@dataclass(frozen=True, slots=True)
class ExportJob:
    automatic: bool
    source_path: str
    output_path: str
    export_base: ExportBase
    regions: list[dict]
    config: dict
    paint_overlay: Optional[np.ndarray] = None
    stamp_overlay: Optional[np.ndarray] = None
    # 可编辑贴片记录：整页预合成放到导出 worker（execute_export_job）内进行，
    # 避免在 GUI 线程分配整页大缓冲
    paste_overlays: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.automatic, bool):
            raise TypeError("automatic export flag must be bool")
        if not self.source_path or not self.output_path:
            raise ValueError("export source and output paths are required")
        if not isinstance(self.export_base, ExportBase):
            raise TypeError("export_base must be an ExportBase")
        if not isinstance(self.regions, list) or not all(
            isinstance(region, dict) for region in self.regions
        ):
            raise TypeError("export regions must be a list of dictionaries")
        if not isinstance(self.config, dict):
            raise TypeError("export config must be a dictionary")

        regions = copy.deepcopy(self.regions)
        config = copy.deepcopy(self.config)
        config.setdefault("render", {})["disable_auto_wrap"] = True
        config.setdefault("upscale", {})["upscale_ratio"] = None
        config.setdefault("colorizer", {})["colorizer"] = "none"
        overlays = {}
        for field_name in ("paint_overlay", "stamp_overlay"):
            overlay = getattr(self, field_name)
            if overlay is None:
                overlays[field_name] = None
                continue
            array = np.asarray(overlay)
            if array.ndim != 3 or array.shape[2] != 4:
                overlays[field_name] = None
                continue
            if not np.any(array[..., 3]):
                overlays[field_name] = None
                continue
            owned = np.array(array, copy=True)
            owned.setflags(write=False)
            overlays[field_name] = owned

        paste_overlays = tuple(
            copy.deepcopy(item)
            for item in self.paste_overlays
            if isinstance(item, dict)
        )
        object.__setattr__(self, "paste_overlays", paste_overlays)

        source_image = image_like_to_pil(self.export_base.source_image)
        if source_image is None:
            raise ValueError("export source image is empty")
        render_image = source_image
        try:
            if self.export_base.kind == "paired":
                render_image = image_like_to_rgb_array(
                    self.export_base.render_image,
                    copy=True,
                )
                if render_image is None:
                    raise ValueError("paired export render image is empty")
            export_base = ExportBase(
                self.export_base.kind,
                source_image,
                render_image,
                self.export_base.mask,
                self.export_base.inpaint_key,
            )
        except Exception:
            _close_images(source_image, render_image)
            raise

        object.__setattr__(self, "export_base", export_base)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "config", config)
        for field_name, overlay in overlays.items():
            object.__setattr__(self, field_name, overlay)

    @property
    def source_key(self) -> str:
        return os.path.normcase(os.path.abspath(self.source_path))

    def release_resources(self) -> None:
        _close_images(
            self.export_base.source_image,
            self.export_base.render_image,
        )


@dataclass(frozen=True, slots=True)
class ExportOutcome:
    automatic: bool
    source_path: str
    output_path: str
    success: bool
    error: Optional[str] = None
    generated_artifact: Optional[InpaintArtifact] = None


class EditorControllerExportService:
    """编辑器工程保存与渲染图片导出流程。"""

    def __init__(self, controller: "EditorController"):
        self.controller = controller
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="editor-export",
        )
        self._state_lock = threading.RLock()
        self._accepting = True
        self._unfinished = 0
        self._pending_auto: dict[str, concurrent.futures.Future] = {}
        # the open page: its key, the stamp of its JSON when read or last saved here, and the
        # history revision its output image was last rendered from (None: not since a failure)
        self._document_key: Optional[str] = None
        self._json_stamp = _UNKNOWN
        self._exported_revision: Optional[int] = None
        # the open page was saved while its repair was still running (its key): the finished
        # repair is written when it arrives
        self._inpaint_save_pending: Optional[str] = None
        from services.export_service import ExportService

        self._export_service = ExportService()

    @property
    def model(self):
        return self.controller.model

    @property
    def logger(self):
        return self.controller.logger

    @property
    def config_service(self):
        return self.controller.config_service

    def has_unsaved_changes(self) -> bool:
        """脏检测唯一真相源：仅成功保存工程数据后标记 clean。"""
        return not self.controller.history_service.is_clean()

    @staticmethod
    def _key(path) -> Optional[str]:
        return os.path.normcase(os.path.abspath(path)) if path else None

    def _revision(self) -> int:
        current = getattr(self.controller.history_service, "current_revision", None)
        return int(current()) if callable(current) else 0

    def _is_open_document(self, source_path) -> bool:
        key = self._key(source_path)
        return bool(key) and key == self._document_key and key == self._key(
            self.model.get_source_image_path()
        )

    def begin_document(self, snapshot) -> None:
        """A page was loaded: its output image counts as up to date, and the stamp of its JSON is
        kept, to notice a write by another program before the editor saves over it."""
        self._document_key = self._key(getattr(snapshot, "source_path", None))
        stamps = dict(getattr(snapshot, "files_signature", ()) or ())
        self._json_stamp = stamps["json"] if "json" in stamps else _UNKNOWN
        self._exported_revision = self._revision()

    def end_document(self) -> None:
        self._document_key = None
        self._json_stamp = _UNKNOWN
        self._exported_revision = None
        self._inpaint_save_pending = None

    def inpaint_installed(self) -> None:
        """A repair finished for the open page after it was saved with the unfinished one: the
        saved inpainted image is older than the saved mask, so the finished one is written now.
        While the page has unsaved edits the next save writes it."""
        source_path = self.model.get_source_image_path()
        if not source_path or self._inpaint_save_pending != self._key(source_path):
            return
        if self.has_unsaved_changes():
            return
        artifact = self.model.get_ready_inpaint_artifact()
        if artifact is None:
            return
        try:
            config = self._build_config_dict(self.config_service.get_config())
            self._export_service.save_inpainted_image(
                os.path.abspath(source_path), artifact.image, config
            )
            self._inpaint_save_pending = None
        except Exception as e:
            self.logger.warning(f"Writing the finished repair of {source_path} failed: {e}")

    def needs_export(self) -> bool:
        """The open page changed since its output image was last rendered (saved with Ctrl+S but
        not exported, or its last export failed)."""
        if self._document_key is None:
            return False
        if self._key(self.model.get_source_image_path()) != self._document_key:
            return False
        return self._exported_revision != self._revision()

    def export_failed(self, source_path) -> None:
        """An export of the open page failed: it is exported again when the page is left."""
        if self._is_open_document(source_path):
            self._exported_revision = None

    def json_changed_elsewhere(self, source_path) -> bool:
        """The page's JSON was written, created or removed by another program (an AI Agent run, the
        batch management) since the editor read or saved it."""
        if self._json_stamp is _UNKNOWN or not self._is_open_document(source_path):
            return False
        from .document_files import json_stamp

        return json_stamp(source_path) != self._json_stamp

    def _ask_changed_on_disk_action(self, source_path: str) -> str:
        """覆盖保存 / 重新载入 / 取消 -> overwrite / reload / cancel."""
        from PyQt6.QtWidgets import QApplication
        from qfluentwidgets import Dialog, PushButton

        view = getattr(self.controller, "view", None)
        dialog = Dialog(
            "工程文件已被改动",
            f"{os.path.basename(source_path)} 的工程文件在编辑器打开后被其他程序改过"
            "（例如 AI Agent 或批量管理）。\n\n"
            "覆盖保存：用编辑器里的内容保存，盖掉那些改动。\n"
            "重新载入：放弃编辑器里还没保存的修改，载入新的内容。",
            view if view is not None else QApplication.activeWindow(),
        )
        dialog.setTitleBarVisible(True)
        dialog.yesButton.setText("覆盖保存")
        dialog.cancelButton.setText("取消")
        reload_button = PushButton("重新载入", dialog.buttonGroup)
        dialog.buttonLayout.insertWidget(1, reload_button, 1)
        chosen = {"value": "overwrite"}
        reload_button.clicked.connect(lambda: chosen.update(value="reload"))
        reload_button.clicked.connect(dialog.accept)
        dialog.setFixedSize(max(dialog.width(), 520), max(dialog.height(), 240))
        if dialog.exec() != Dialog.DialogCode.Accepted:
            return "cancel"
        return chosen["value"]

    def save_editor_state(self) -> bool:
        """同步保存 JSON 与当前内存快照中的修复图，不等待后台修复。"""
        self.controller.commit_pending_edits()
        source_path = self.model.get_source_image_path()
        if not source_path:
            self._reject_export("保存失败：当前图片没有来源路径")
            return False
        if self.json_changed_elsewhere(source_path):
            action = self._ask_changed_on_disk_action(source_path)
            if action == "reload":
                self.controller.document_service.do_load_image(source_path)
                return False
            if action != "overwrite":
                self._reject_export("没有保存：这一页的工程文件被其他程序改过")
                return False

        try:
            image = self.model.get_image()
            if image is None:
                self._reject_export("保存失败：缺少图像数据")
                return False
            regions = self.model.get_regions() or []
            mask = self.model.get_refined_mask()
            if mask is None:
                mask = self.model.get_raw_mask()

            inpainted_image = None
            if mask is not None and np.any(mask):
                display_layers = self.model.get_display_layers()
                if display_layers is not None:
                    inpainted_image = display_layers.inpaint_display_image

            source_path = os.path.abspath(source_path)
            config = self._build_config_dict(self.config_service.get_config())
            self._export_service.save_editor_project(
                source_path,
                regions,
                mask,
                config,
                paint_overlay=self.model.get_paint_overlay_image(),
                stamp_overlay=self.model.get_stamp_overlay_image(),
                paste_overlays=self.model.get_paste_overlays() or [],
            )
            if inpainted_image is not None:
                self._export_service.save_inpainted_image(
                    source_path, inpainted_image, config
                )
            else:
                self._export_service.delete_inpainted_image(source_path)
            self.controller.history_service.mark_clean()
            unfinished = False
            if inpainted_image is not None:
                export_base = self.model.get_export_base()
                unfinished = getattr(export_base, "kind", None) == "backend_inpaint"
            self._inpaint_save_pending = self._key(source_path) if unfinished else None
            if self._is_open_document(source_path):
                from .document_files import json_stamp

                self._json_stamp = json_stamp(source_path)
            toast_manager = self.controller.get_toast_manager()
            if toast_manager is not None:
                toast_manager.show_success("保存成功", 2500)
            return True
        except Exception as e:
            self.logger.error("Failed to save editor state", exc_info=True)
            self._reject_export(f"保存失败：{e}")
            return False

    def export_image(
        self,
        automatic: bool = False,
    ) -> Optional[concurrent.futures.Future]:
        source_path = self.model.get_source_image_path()
        if not source_path:
            return self._reject_export("导出失败：当前图片没有来源路径")
        source_path = os.path.abspath(source_path)

        try:
            export_base = self.model.get_export_base()
            if export_base is None:
                return self._reject_export("导出失败：缺少活动文档")
            config = self._build_config_dict(self.config_service.get_config())

            # 贴片整页预合成移到导出 worker（ExportService.execute_export_job）内执行，
            # GUI 线程只负责把规范化记录快照塞进 job，避免大图卡界面
            job = ExportJob(
                automatic=bool(automatic),
                source_path=source_path,
                output_path=self._export_service.build_output_path(config, source_path),
                export_base=export_base,
                regions=self._build_enhanced_regions(self.model.get_regions() or []),
                config=config,
                paint_overlay=self.model.get_paint_overlay_image(),
                stamp_overlay=self.model.get_stamp_overlay_image(),
                paste_overlays=tuple(self.model.get_paste_overlays() or ()),
            )
            future = self._submit_job(job)
            if future is None:
                job.release_resources()
                return self._reject_export("导出队列已经关闭")
            if self._is_open_document(source_path):
                self._exported_revision = self._revision()
            return future
        except Exception as e:
            self.logger.error(f"Error during export request: {e}", exc_info=True)
            return self._reject_export(f"导出快照创建失败：{e}")

    def _reject_export(self, message: str):
        toast_manager = self.controller.get_toast_manager()
        if toast_manager is not None:
            toast_manager.show_error(message, 5000)
        return None

    def _submit_job(self, job: ExportJob) -> Optional[concurrent.futures.Future]:
        source_key = job.source_key if job.automatic else None
        with self._state_lock:
            if not self._accepting:
                return None
            previous = self._pending_auto.get(source_key) if source_key else None
            try:
                future = self._executor.submit(self.execute_export_job, job)
            except RuntimeError:
                return None
            self._unfinished += 1
            if source_key:
                self._pending_auto[source_key] = future

        future.add_done_callback(
            lambda done, queued_job=job, key=source_key: self._on_job_done(
                done,
                queued_job,
                key,
            )
        )
        if previous is not None:
            previous.cancel()
        self.controller._export_queue_status_signal.emit(self.unfinished_count())
        return future

    def _on_job_done(
        self,
        future: concurrent.futures.Future,
        job: ExportJob,
        source_key: Optional[str],
    ) -> None:
        outcome = None
        if not future.cancelled():
            try:
                outcome = future.result()
            except Exception as e:
                self.logger.exception("Unhandled editor export error")
                outcome = ExportOutcome(
                    automatic=job.automatic,
                    source_path=job.source_path,
                    output_path=job.output_path,
                    success=False,
                    error=str(e),
                )

        with self._state_lock:
            self._unfinished = max(0, self._unfinished - 1)
            if source_key and self._pending_auto.get(source_key) is future:
                self._pending_auto.pop(source_key, None)
            unfinished = self._unfinished

        job.release_resources()
        self.controller._export_queue_status_signal.emit(unfinished)
        if outcome is not None:
            self.controller._export_job_finished_signal.emit(outcome)

    def unfinished_count(self) -> int:
        with self._state_lock:
            return self._unfinished

    def shutdown(self) -> None:
        with self._state_lock:
            if not self._accepting:
                return
            self._accepting = False
        self._executor.shutdown(wait=True, cancel_futures=False)

    @staticmethod
    def resolve_effective_box_local(region: dict):
        if not isinstance(region, dict):
            return None
        region = normalize_region_geometry_data(region)

        custom_box = region.get("white_frame_rect_local")
        render_box = region.get("render_box_rect_local")
        has_custom = bool(region.get("has_custom_white_frame", False))

        # 解绑：与编辑器 snapshot 同步——用户手动白框存在时优先白框，
        # 让导出和预览的渲染中心走同一条路。
        if (
            has_custom
            and isinstance(custom_box, (list, tuple))
            and len(custom_box) == 4
        ):
            return custom_box
        if isinstance(render_box, (list, tuple)) and len(render_box) == 4:
            return render_box
        if isinstance(custom_box, (list, tuple)) and len(custom_box) == 4:
            return custom_box
        return None

    @classmethod
    def apply_white_frame_center(cls, region: dict) -> None:
        """将 center 重算为白框世界中心，并同步平移 local 坐标以免漂移。"""
        wf_local = cls.resolve_effective_box_local(region)
        if not (isinstance(wf_local, (list, tuple)) and len(wf_local) == 4):
            return

        base_center = region.get("center")
        if not (
            isinstance(base_center, (list, tuple, np.ndarray)) and len(base_center) >= 2
        ):
            # 旧工程通常不保存 center；必须与编辑器快照使用同一套 lines
            # 回退中心，否则白框局部偏移不会进入导出，文字会沿旋转轴漂移。
            try:
                base_center = RegionGeometryState.from_region_data(region).center
            except (TypeError, ValueError, IndexError):
                return
        try:
            left, top, right, bottom = (float(v) for v in wf_local)
            lx = (left + right) / 2.0
            ly = (top + bottom) / 2.0
            cx, cy = float(base_center[0]), float(base_center[1])
            angle = float(region.get("angle") or 0.0)
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            region["center"] = [
                cx + lx * cos_a - ly * sin_a,
                cy + lx * sin_a + ly * cos_a,
            ]
            # 同步平移 local 坐标，以新 center 为原点，防止存/读漂移
            if "white_frame_rect_local" in region:
                wf = region["white_frame_rect_local"]
                if isinstance(wf, (list, tuple)) and len(wf) == 4:
                    region["white_frame_rect_local"] = [
                        float(wf[0]) - lx,
                        float(wf[1]) - ly,
                        float(wf[2]) - lx,
                        float(wf[3]) - ly,
                    ]
            if "render_box_rect_local" in region:
                rb = region["render_box_rect_local"]
                if isinstance(rb, (list, tuple)) and len(rb) == 4:
                    region["render_box_rect_local"] = [
                        float(rb[0]) - lx,
                        float(rb[1]) - ly,
                        float(rb[2]) - lx,
                        float(rb[3]) - ly,
                    ]
        except (TypeError, ValueError):
            return

    @staticmethod
    def _build_config_dict(config) -> dict:
        if hasattr(config, "model_dump"):
            return config.model_dump()
        if hasattr(config, "dict"):
            return config.dict()
        return {}

    def _build_enhanced_regions(self, regions: list[dict]) -> list[dict]:
        render_service = get_render_parameter_service()
        enhanced_regions = []
        for index, region in enumerate(regions):
            enhanced_region = region.copy()

            self.apply_white_frame_center(enhanced_region)
            enhanced_region.update(
                render_service.export_parameters_for_backend(index, enhanced_region)
            )
            enhanced_regions.append(enhanced_region)
        return enhanced_regions

    def execute_export_job(self, job: ExportJob) -> ExportOutcome:
        """Execute the single immutable job accepted by the serial worker."""
        try:
            result = self._export_service.execute_export_job(job)
        except Exception as error:
            self.logger.error("Failed to render editor export job", exc_info=True)
            return ExportOutcome(
                automatic=job.automatic,
                source_path=job.source_path,
                output_path=job.output_path,
                success=False,
                error=str(error),
            )

        if result.error is not None:
            return ExportOutcome(
                automatic=job.automatic,
                source_path=job.source_path,
                output_path=job.output_path,
                success=False,
                error=result.error,
            )
        generated_artifact = None
        if result.generated_inpainted_image is not None:
            generated_artifact = InpaintArtifact(
                job.export_base.inpaint_key,
                job.export_base.mask,
                result.generated_inpainted_image,
            )
        return ExportOutcome(
            automatic=job.automatic,
            source_path=job.source_path,
            output_path=job.output_path,
            success=True,
            generated_artifact=generated_artifact,
        )
