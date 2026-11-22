"""
压缩包/文档格式图片提取工具
支持 PDF、EPUB、CBZ 格式
"""
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from typing import List, Optional, Tuple

from manga_translator.image_formats import SUPPORTED_IMAGE_EXTENSIONS

# 支持的压缩包/文档格式
ARCHIVE_EXTENSIONS = {'.pdf', '.epub', '.cbz', '.cbr', '.zip'}

# 支持的图片格式
IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS

ORIGINAL_IMAGE_DIRNAME = 'original_images'
WORK_DIRNAME = 'manga_translator_work'  # path_manager.WORK_DIR_NAME: the pages' work files, beside them
REPLACED_DIRNAME = 'replaced'
"""Under the work folder: pages an archive's new version no longer has, with their work files."""

logger = logging.getLogger(__name__)
ARCHIVE_SOURCE_MARKER_FILENAME = '.archive_source.txt'
EXTRACT_META_FILENAME = '.extract_meta.json'


def is_archive_file(file_path: str) -> bool:
    """检查文件是否是支持的压缩包/文档格式"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ARCHIVE_EXTENSIONS


def get_output_extract_dir(output_base_dir: str, archive_path: str) -> str:
    """获取解压到输出目录下的目录：<输出目录>/<文件名>/original_images"""
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    return os.path.join(output_base_dir, archive_name, ORIGINAL_IMAGE_DIRNAME)

def get_output_extract_root(output_base_dir: str, archive_path: str) -> str:
    """获取解压根目录：<输出目录>/<文件名>"""
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    return os.path.join(output_base_dir, archive_name)

def get_output_extract_marker_path(output_base_dir: str, archive_path: str) -> str:
    """获取压缩包来源标记文件路径。"""
    return os.path.join(
        get_output_extract_root(output_base_dir, archive_path),
        ARCHIVE_SOURCE_MARKER_FILENAME
    )

def _normalize_abs_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))

def _build_extract_meta(archive_path: str) -> dict:
    return {
        'archive_path': _normalize_abs_path(archive_path),
        'archive_mtime': int(os.path.getmtime(archive_path)) if os.path.exists(archive_path) else 0,
        'archive_size': int(os.path.getsize(archive_path)) if os.path.exists(archive_path) else 0,
    }

def _get_extract_meta_path(output_dir: str) -> str:
    return os.path.join(output_dir, EXTRACT_META_FILENAME)

def _read_extract_meta(output_dir: str) -> Optional[dict]:
    meta_path = _get_extract_meta_path(output_dir)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None

def _write_extract_meta(output_dir: str, archive_path: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta_path = _get_extract_meta_path(output_dir)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(_build_extract_meta(archive_path), f, ensure_ascii=False, indent=2)

def _clear_extract_output_dir(output_dir: str) -> None:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

def check_output_extract_conflict(output_base_dir: str, archive_path: str) -> bool:
    """
    检查同名解压目录是否和当前压缩包冲突。
    True 表示存在冲突（同名目录但来源不是当前 archive_path）。
    """
    root_dir = get_output_extract_root(output_base_dir, archive_path)
    if not os.path.isdir(root_dir):
        return False

    marker_path = get_output_extract_marker_path(output_base_dir, archive_path)
    if not os.path.exists(marker_path):
        # 兼容旧版本：尝试读取解压目录元数据判断来源
        extract_dir = get_output_extract_dir(output_base_dir, archive_path)
        cached_meta = _read_extract_meta(extract_dir)
        if cached_meta and cached_meta.get('archive_path') == _normalize_abs_path(archive_path):
            return False
        # 没有可用元数据时保守视为冲突，避免误复用同名目录
        return True

    try:
        with open(marker_path, 'r', encoding='utf-8') as f:
            recorded_source = f.read().strip()
    except Exception:
        return True

    if not recorded_source:
        return True

    return _normalize_abs_path(recorded_source) != _normalize_abs_path(archive_path)

def clear_output_extract_root(output_base_dir: str, archive_path: str) -> None:
    """删除同名解压根目录（用于覆盖模式下的冲突处理）。"""
    root_dir = get_output_extract_root(output_base_dir, archive_path)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir, ignore_errors=True)

def write_output_extract_marker(output_base_dir: str, archive_path: str) -> None:
    """写入压缩包来源标记，用于识别同名目录冲突。"""
    marker_path = get_output_extract_marker_path(output_base_dir, archive_path)
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, 'w', encoding='utf-8') as f:
        f.write(_normalize_abs_path(archive_path))


def get_temp_extract_dir(archive_path: str) -> str:
    """获取压缩包的临时解压目录"""
    # 使用系统临时目录下的固定子目录，便于管理
    base_temp = os.path.join(tempfile.gettempdir(), 'manga_translator_archives')
    os.makedirs(base_temp, exist_ok=True)
    
    # 使用文件名和修改时间生成唯一目录名
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    mtime = int(os.path.getmtime(archive_path)) if os.path.exists(archive_path) else 0
    unique_name = f"{archive_name}_{mtime}"
    
    return os.path.join(base_temp, unique_name)


PDF_PAGE_COVER = 0.95
"""The share of the page an embedded image must cover to be taken as the page."""
PDF_MIN_ZOOM, PDF_MAX_ZOOM = 2.0, 5.0
"""How much a rendered page is enlarged: about its images' own resolution, within these."""


def _pdf_visible_text(page) -> bool:
    """The page shows text (an OCR layer of a scan is drawn invisible, and does not count)."""
    try:
        for span in page.get_texttrace():
            if span.get("type") == 3 or float(span.get("opacity", 1.0) or 0) <= 0:
                continue
            if any(chr(char[0]).strip() for char in span.get("chars", ()) if char and char[0] > 0):
                return True
        return False
    except Exception:
        return bool(page.get_text("text").strip())


def _pdf_page_image(doc, page) -> Optional[dict]:
    """The page's embedded image when it is the whole page (``doc.extract_image``); None when the
    page must be rendered: text or vector drawings on it, several images, or one that is turned,
    masked or does not cover the page."""
    import fitz

    images = page.get_images(full=True)
    if len(images) != 1 or images[0][1]:  # several images, or one with a soft mask
        return None
    xref = images[0][0]
    try:
        infos = [info for info in page.get_image_info(xrefs=True) if info.get("xref") == xref]
    except Exception:
        return None
    if len(infos) != 1 or infos[0].get("has-mask"):
        return None
    a, b, c, d = (infos[0].get("transform") or (0, 0, 0, 0))[:4]
    if abs(b) > 1e-6 or abs(c) > 1e-6 or a <= 0 or d <= 0:  # turned, slanted or mirrored
        return None
    shown = fitz.Rect(infos[0]["bbox"]) & page.rect
    if shown.is_empty or shown.get_area() < PDF_PAGE_COVER * page.rect.get_area():
        return None
    if _pdf_visible_text(page):
        return None
    try:
        if page.get_drawings():
            return None
    except Exception:
        return None
    try:
        return doc.extract_image(xref)
    except Exception:
        return None


def _pdf_zoom(page) -> float:
    """About the resolution of the page's images (a scan keeps its pixels), within the limits."""
    scales = []
    try:
        for info in page.get_image_info():
            x0, y0, x1, y1 = (float(v) for v in info["bbox"])
            shown = (x1 - x0) * (y1 - y0)
            pixels = float(info.get("width") or 0) * float(info.get("height") or 0)
            if shown > 0 and pixels > 0:  # by area: the same for a turned image
                scales.append((pixels / shown) ** 0.5)
    except Exception:
        pass
    return min(PDF_MAX_ZOOM, max([PDF_MIN_ZOOM] + scales))


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[str]:
    """从 PDF 文件中提取每一页的图片：整页就是一张内嵌原图时直接取原图（不损失画质），
    其他页（有可见文字、矢量图形、多张图、旋转或带遮罩的图）整页渲染，避免漏掉排成文字的对白。"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("需要安装 PyMuPDF: pip install PyMuPDF")

    os.makedirs(output_dir, exist_ok=True)
    extracted_images = []
    img_count = 0

    doc = None
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            base = _pdf_page_image(doc, page)
            if base is not None:
                img_count += 1
                image_path = os.path.join(output_dir, f"page_{img_count:04d}.{base['ext']}")
                with open(image_path, 'wb') as f:
                    f.write(base['image'])
                extracted_images.append(image_path)
                continue
            try:
                zoom = _pdf_zoom(page)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                img_count += 1
                image_path = os.path.join(output_dir, f"page_{img_count:04d}.png")
                pix.save(image_path)
                extracted_images.append(image_path)
                pix = None
            except Exception:
                pass
    finally:
        if doc is not None:
            doc.close()

    return sorted(extracted_images)


def extract_images_from_epub(epub_path: str, output_dir: str) -> List[str]:
    """
    从 EPUB 文件中按书籍实际阅读顺序提取图片。
    1. 解析 EPUB 清单文件（<spine> + <manifest>），保证严格按阅读顺序排列；
    2. 优先直接提取每页引用的原始高清图片（零损耗保留原图分辨率与格式）；
    3. 若页面为纯文字/SVG/无独立原图页，回退使用 PyMuPDF (fitz) 渲染当前页为 PNG；
    4. 统一命名为 page_{count:04d}.{ext}，与 PDF 处理逻辑完全对齐；
    5. 异常情况自动兜底处理，确保绝对不漏页、不乱序。
    """
    os.makedirs(output_dir, exist_ok=True)
    extracted_images = []
    img_count = 0

    fitz_doc = None
    try:
        import fitz
        try:
            fitz_doc = fitz.open(epub_path)
        except Exception:
            fitz_doc = None
    except ImportError:
        fitz_doc = None

    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            namelist = zf.namelist()
            lower_map = {name.lower(): name for name in namelist}

            # 1. 定位 OPF 清单路径
            opf_path = None
            try:
                container = ET.fromstring(zf.read('META-INF/container.xml'))
                rootfile = container.find('.//{*}rootfile')
                if rootfile is not None and rootfile.get('full-path'):
                    fp = rootfile.get('full-path')
                    opf_path = lower_map.get(fp.lower(), fp)
            except Exception:
                pass

            if not opf_path:
                opf_path = next((name for name in namelist if name.lower().endswith('.opf')), None)

            # 2. 解析 OPF 清单与阅读顺序
            ordered_targets = []
            if opf_path and opf_path in namelist:
                try:
                    opf_dir = posixpath.dirname(opf_path)
                    opf = ET.fromstring(zf.read(opf_path))
                    manifest = {
                        item.get('id'): item.get('href', '')
                        for item in opf.findall('.//{*}item')
                        if item.get('id')
                    }
                    spine_ids = [ref.get('idref') for ref in opf.findall('.//{*}itemref') if ref.get('idref')]

                    img_pattern = re.compile(
                        r'(?:src|href)=["\']([^"\']+\.(?:' + '|'.join(e.lstrip('.') for e in IMAGE_EXTENSIONS) + r'))',
                        re.IGNORECASE
                    )

                    seen_in_spine = set()
                    for sid in spine_ids:
                        href = manifest.get(sid)
                        if not href:
                            continue
                        target = posixpath.normpath(posixpath.join(opf_dir, urllib.parse.unquote(href)))
                        ext = os.path.splitext(target)[1].lower()

                        if ext in IMAGE_EXTENSIONS:
                            real_p = lower_map.get(target.lower())
                            if real_p and real_p not in seen_in_spine:
                                seen_in_spine.add(real_p)
                                ordered_targets.append((real_p, None))
                        else:
                            real_html = lower_map.get(target.lower())
                            found = False
                            if real_html:
                                html_text = zf.read(real_html).decode('utf-8', errors='replace')
                                for match in img_pattern.findall(html_text):
                                    img_ref = urllib.parse.unquote(match.split('?')[0].split('#')[0])
                                    img_full = posixpath.normpath(posixpath.join(posixpath.dirname(real_html), img_ref))
                                    real_img = lower_map.get(img_full.lower())
                                    if real_img and real_img not in seen_in_spine:
                                        seen_in_spine.add(real_img)
                                        ordered_targets.append((real_img, None))
                                        found = True
                            if not found:
                                # 纯文本/SVG/无独立原图页，记录其在 spine 中的页码以供 fitz 渲染
                                ordered_targets.append((None, len(ordered_targets)))
                except Exception:
                    ordered_targets.clear()

            # 3. 按阅读顺序提取原始高清图片（或回退渲染）
            extracted_paths = set()
            for zip_rel, spine_page_idx in ordered_targets:
                if zip_rel:
                    ext = os.path.splitext(zip_rel)[1].lower() or '.png'
                    img_count += 1
                    out_path = os.path.join(output_dir, f"page_{img_count:04d}{ext}")
                    with zf.open(zip_rel) as src, open(out_path, 'wb') as dst:
                        dst.write(src.read())
                    extracted_images.append(out_path)
                    extracted_paths.add(zip_rel)
                elif fitz_doc is not None and spine_page_idx is not None and spine_page_idx < len(fitz_doc):
                    try:
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = fitz_doc[spine_page_idx].get_pixmap(matrix=mat)
                        img_count += 1
                        out_path = os.path.join(output_dir, f"page_{img_count:04d}.png")
                        pix.save(out_path)
                        extracted_images.append(out_path)
                    except Exception:
                        pass

            # 4. 追加未在 spine 显式引用的剩余图片（确保绝对不漏页）
            remaining = [
                name for name in namelist
                if name not in extracted_paths and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            ]
            remaining.sort(key=natural_sort_key)
            for rem in remaining:
                ext = os.path.splitext(rem)[1].lower() or '.png'
                img_count += 1
                out_path = os.path.join(output_dir, f"page_{img_count:04d}{ext}")
                with zf.open(rem) as src, open(out_path, 'wb') as dst:
                    dst.write(src.read())
                extracted_images.append(out_path)

            # 5. 若未提取出任何图片，通过 fitz 全书渲染兜底
            if not extracted_images and fitz_doc is not None:
                for fitz_page in fitz_doc:
                    try:
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = fitz_page.get_pixmap(matrix=mat)
                        img_count += 1
                        out_path = os.path.join(output_dir, f"page_{img_count:04d}.png")
                        pix.save(out_path)
                        extracted_images.append(out_path)
                    except Exception:
                        pass
    finally:
        if fitz_doc is not None:
            fitz_doc.close()

    return sorted(extracted_images)


def extract_images_from_cbz(cbz_path: str, output_dir: str) -> List[str]:
    """从 CBZ (Comic Book ZIP) 文件中提取图片"""
    os.makedirs(output_dir, exist_ok=True)
    extracted_images = []
    
    with zipfile.ZipFile(cbz_path, 'r') as zf:
        # 获取所有图片文件并排序
        image_files = []
        for file_info in zf.infolist():
            if file_info.is_dir():
                continue
            ext = os.path.splitext(file_info.filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(file_info)
        
        # 按文件名自然排序
        image_files.sort(key=lambda x: natural_sort_key(x.filename))
        
        for idx, file_info in enumerate(image_files):
            base_name = os.path.basename(file_info.filename)
            # 添加序号前缀以保持顺序
            new_name = f"{idx:04d}_{base_name}"
            output_path = os.path.join(output_dir, new_name)
            
            with zf.open(file_info) as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
            extracted_images.append(output_path)
    
    return extracted_images


def extract_images_from_cbr(cbr_path: str, output_dir: str) -> List[str]:
    """从 CBR (Comic Book RAR) 文件中提取图片"""
    try:
        import rarfile
    except ImportError:
        raise ImportError("需要安装 rarfile: pip install rarfile")
    
    os.makedirs(output_dir, exist_ok=True)
    extracted_images = []
    
    with rarfile.RarFile(cbr_path, 'r') as rf:
        image_files = []
        for file_info in rf.infolist():
            if file_info.is_dir():
                continue
            ext = os.path.splitext(file_info.filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(file_info)
        
        image_files.sort(key=lambda x: natural_sort_key(x.filename))
        
        for idx, file_info in enumerate(image_files):
            base_name = os.path.basename(file_info.filename)
            new_name = f"{idx:04d}_{base_name}"
            output_path = os.path.join(output_dir, new_name)
            
            with rf.open(file_info) as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
            extracted_images.append(output_path)
    
    return extracted_images


def natural_sort_key(s: str):
    """自然排序键，支持数字排序"""
    import re
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]


def extract_images_from_archive(archive_path: str, output_dir: Optional[str] = None) -> Tuple[List[str], str]:
    """
    从压缩包/文档中提取图片
    
    Args:
        archive_path: 压缩包/文档路径
        output_dir: 输出目录，如果为 None 则使用临时目录
    
    Returns:
        (提取的图片路径列表, 输出目录)
    """
    if output_dir is None:
        output_dir = get_temp_extract_dir(archive_path)

    expected_meta = _build_extract_meta(archive_path)

    # 如果目录已存在且缓存元数据一致，直接返回缓存结果
    if os.path.exists(output_dir):
        existing_images = []
        for f in os.listdir(output_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                existing_images.append(os.path.join(output_dir, f))
        cached_meta = _read_extract_meta(output_dir)
        if existing_images and cached_meta == expected_meta:
            return sorted(existing_images), output_dir
        if os.path.isdir(os.path.join(output_dir, WORK_DIRNAME)):
            # 工作目录（校稿 JSON、手修图、Agent 备份）在里面：先解压到旁边，再按内容核对每一页
            images = _update_extract_output_dir(archive_path, output_dir)
            _write_extract_meta(output_dir, archive_path)
            return images, output_dir
        # 目录存在但缓存不可用（来源/版本不匹配或残留脏数据），没有工作文件，清空后重解压
        _clear_extract_output_dir(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    images = _extract_to(archive_path, output_dir)
    _write_extract_meta(output_dir, archive_path)
    return images, output_dir


def _extract_to(archive_path: str, output_dir: str) -> List[str]:
    ext = os.path.splitext(archive_path)[1].lower()
    if ext == '.pdf':
        return extract_images_from_pdf(archive_path, output_dir)
    if ext == '.epub':
        return extract_images_from_epub(archive_path, output_dir)
    if ext in {'.cbz', '.zip'}:
        return extract_images_from_cbz(archive_path, output_dir)
    if ext == '.cbr':
        return extract_images_from_cbr(archive_path, output_dir)
    raise ValueError(f"不支持的文件格式: {ext}")


def _digest(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _page_work_files(work_dir: str, image_name: str) -> List[str]:
    """The work files of one page (paths relative to ``work_dir``), by the names ``path_manager``
    gives them."""
    stem, ext = os.path.splitext(image_name)
    found = []
    for sub, name in (('json', f'{stem}_translations.json'), ('inpainted', f'{stem}_inpainted{ext}'),
                      ('paint_overlay', f'{stem}_overlay.png'), ('editor_base', image_name),
                      ('yolo_labels', f'{stem}.txt'), (os.path.join('agent', 'pages'), f'{stem}.agent.json')):
        if os.path.isfile(os.path.join(work_dir, sub, name)):
            found.append(os.path.join(sub, name))
    for sub, suffix in (('originals', '_original.'), ('translations', '_translated.')):
        folder = os.path.join(work_dir, sub)
        if os.path.isdir(folder):
            found += [os.path.join(sub, name) for name in sorted(os.listdir(folder))
                      if name.startswith(stem + suffix) and os.path.isfile(os.path.join(folder, name))]
    return found


def _renamed(rel: str, old_name: str, new_name: str) -> str:
    """A work file of ``old_name`` as the one of ``new_name``."""
    folder, name = os.path.split(rel)
    old_stem, new_stem = os.path.splitext(old_name)[0], os.path.splitext(new_name)[0]
    if name == old_name:
        name = new_name
    elif name.startswith(old_stem):
        name = new_stem + name[len(old_stem):]
        if name.endswith('_inpainted' + os.path.splitext(old_name)[1]):  # it keeps the image's extension
            name = name[:-len(os.path.splitext(old_name)[1])] + os.path.splitext(new_name)[1]
    return os.path.join(folder, name)


def _move(source: str, target: str) -> None:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.move(source, target)


def _remove_empty_dirs(folder: str) -> None:
    for root, dirs, files in os.walk(folder, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(root)
            except OSError:
                pass


def _remap_agent_records(work_dir: str, renamed: dict, gone: set) -> None:
    """The AI Agent's records that are kept by page name follow the pages (``renamed``: old stem ->
    new stem); the records of pages that are gone are dropped (their sidecars went aside with them)."""
    if not renamed and not gone:
        return
    agent_dir = os.path.join(work_dir, 'agent')

    def remap(data: dict) -> dict:
        result = {key: value for key, value in data.items() if key not in gone and key not in renamed}
        result.update({renamed[key]: value for key, value in data.items() if key in renamed})
        return result

    for name, nested in (('saved_json.json', False), ('saved_json.stat.json', False),
                         ('saved_inpainted.json', False), ('pending_steps.json', True)):
        path = os.path.join(agent_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            if nested:
                data = {step: remap(pages) if isinstance(pages, dict) else pages for step, pages in data.items()}
            else:
                data = remap(data)
            temp = path + '.tmp'
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp, path)
        except Exception:
            logger.warning(f"could not update {path} for the archive's new pages", exc_info=True)


def _update_extract_output_dir(archive_path: str, output_dir: str) -> List[str]:
    """Bring an extraction folder that has work files (校稿 JSON, the user's repairs, the AI Agent's
    backups) up to the archive's new version, without deleting anything; returns its pages.

    The new version is extracted beside the folder first, and each page is matched by its content:
    an unchanged page stays as it is; a page that only got another name (a page added in front
    shifts every ``0000_`` prefix) takes its work files and the AI Agent's records along; a page
    that is gone, or whose name now has other content, is moved with its work files to
    ``manga_translator_work/replaced/<time>/``."""
    staging = tempfile.mkdtemp(prefix='.extract-', dir=os.path.dirname(os.path.abspath(output_dir)))
    try:
        extracted = _extract_to(archive_path, staging)
        new = {os.path.relpath(p, staging): p for p in extracted}
        old = {name: os.path.join(output_dir, name) for name in os.listdir(output_dir)
               if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
               and os.path.isfile(os.path.join(output_dir, name))}
        new_hash = {name: _digest(path) for name, path in new.items()}
        old_hash = {name: _digest(path) for name, path in old.items()}
        unchanged = {name for name in new if name in old and new_hash[name] == old_hash[name]}
        by_hash: dict = {}
        for name in sorted(old):
            if name not in unchanged:
                by_hash.setdefault(old_hash[name], []).append(name)
        moves = {}  # old name -> new name: the same page under another name
        for name in sorted(new):
            if name not in unchanged and by_hash.get(new_hash[name]):
                moves[by_hash[new_hash[name]].pop(0)] = name
        gone = [name for name in sorted(old) if name not in unchanged and name not in moves]

        work_dir = os.path.join(output_dir, WORK_DIRNAME)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        aside = os.path.join(work_dir, REPLACED_DIRNAME, stamp)
        number = 1
        while os.path.exists(aside):
            number += 1
            aside = os.path.join(work_dir, REPLACED_DIRNAME, f'{stamp}~{number}')
        # 1. pages that are gone, or whose name now has other content: aside, with their work files
        for name in gone:
            for rel in _page_work_files(work_dir, name):
                _move(os.path.join(work_dir, rel), os.path.join(aside, rel))
            _move(old[name], os.path.join(aside, ORIGINAL_IMAGE_DIRNAME, name))
        # 2. the same page under another name: its work files go along (through a holding folder, as
        #    the names can chain: 0000_a -> 0001_a, 0001_b -> 0002_b)
        hold = aside + '.moving'
        held = {}
        for old_name in moves:
            held[old_name] = _page_work_files(work_dir, old_name)
            for rel in held[old_name]:
                _move(os.path.join(work_dir, rel), os.path.join(hold, old_name, rel))
        for old_name, new_name in moves.items():
            for rel in held[old_name]:
                _move(os.path.join(hold, old_name, rel), os.path.join(work_dir, _renamed(rel, old_name, new_name)))
            if old_name not in new:  # its bytes are the new name's now
                os.remove(old[old_name])
        if os.path.isdir(hold):
            _remove_empty_dirs(hold)
        # 3. the new version's images that are not there yet
        for name, path in new.items():
            if name not in unchanged:
                _move(path, os.path.join(output_dir, name))
        _remap_agent_records(work_dir, {os.path.splitext(o)[0]: os.path.splitext(n)[0] for o, n in moves.items()},
                             {os.path.splitext(n)[0] for n in gone})
        if moves or gone:
            logger.warning(
                f"{os.path.basename(archive_path)} changed: {len(unchanged)} page(s) unchanged, {len(moves)} "
                f"renamed with their work files, {len(new) - len(unchanged) - len(moves)} new"
                + (f"; {len(gone)} page(s) no longer in it, or replaced, were moved with their work files to "
                   f"{aside}" if gone else ""))
        return sorted(os.path.join(output_dir, name) for name in new)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def cleanup_temp_archives():
    """清理所有临时解压目录"""
    base_temp = os.path.join(tempfile.gettempdir(), 'manga_translator_archives')
    if os.path.exists(base_temp):
        shutil.rmtree(base_temp, ignore_errors=True)


def cleanup_archive_temp(archive_path: str):
    """清理指定压缩包的临时解压目录"""
    temp_dir = get_temp_extract_dir(archive_path)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
