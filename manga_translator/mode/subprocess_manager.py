#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
子进程管理器 - 支持内存管理和断点续传
"""
# import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, 'frozen', False)
    else Path(__file__).resolve().parent.parent.parent
)

# 内存监控阈值
DEFAULT_MEMORY_THRESHOLD_MB = 0  # 默认不限制绝对内存
DEFAULT_MEMORY_THRESHOLD_PERCENT = 80  # 默认达到系统总内存80%时重启
DEFAULT_BATCH_SIZE_PER_RESTART = 50


def get_memory_usage_mb() -> float:
    """获取当前进程的内存使用量（MB）"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0


def get_total_memory_mb() -> float:
    """获取系统总内存（MB）"""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 / 1024
    except ImportError:
        return 0


def get_system_memory_percent() -> float:
    """获取系统总内存使用率（包括所有进程）"""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return 0





def over_memory_limit(memory_limit_mb: int, memory_limit_percent: int) -> Optional[str]:
    """Why this process is over its memory limit (``--memory-limit`` MB of its own, or
    ``--memory-percent`` of the system's memory in use); None when it is not, or no limit is set."""
    if memory_limit_mb > 0:
        used = get_memory_usage_mb()
        if used > memory_limit_mb:
            return f"{used:.0f} MB > {memory_limit_mb} MB"
    if memory_limit_percent > 0:
        percent = get_system_memory_percent()
        if percent > memory_limit_percent:
            return f"{percent:.0f}% > {memory_limit_percent}%"
    return None


def _output_key(file_path: str, input_folders: Iterable[str], output_format: Optional[str]) -> str:
    """Where the translator writes the page (``MangaTranslator._calculate_output_path``), as a key:
    the chapter folder under the output folder, the name and the format."""
    parent = os.path.normpath(os.path.dirname(file_path))
    folder_part = ""
    if os.path.basename(parent) == "original_images":  # an extracted archive: <output>/<archive>/
        folder_part = os.path.join("archive", os.path.basename(os.path.dirname(parent)))
    else:
        for folder in input_folders:
            if parent.startswith(folder):
                relative = os.path.relpath(parent, folder)
                folder_part = os.path.basename(folder) if relative == "." else os.path.join(
                    os.path.basename(folder), relative)
                break
    from manga_translator.utils.path_manager import output_file_name
    name = output_file_name(file_path, output_format)
    return os.path.normcase(os.path.normpath(os.path.join(folder_part, name.lower())))


def same_output_paths(files: Sequence[str], input_folders: Iterable[str],
                      output_format: Optional[str]) -> Dict[str, str]:
    """Pages that would be written where an earlier page of ``files`` is -> that earlier page (two
    chapters' ``001.png`` given as files, or ``001.jpg`` and ``001.png`` saved as PNG)."""
    folders = list(input_folders)
    first: Dict[str, str] = {}
    clashes: Dict[str, str] = {}
    for path in files:
        key = _output_key(path, folders, output_format)
        if key in first:
            clashes[path] = first[key]
        else:
            first[key] = path
    return clashes


def worker_translate_batch(
    file_paths: List[str],
    output_dir: str,
    verbose: bool,
    overwrite: bool,
    config_dict: dict,
    memory_limit_mb: int,
    memory_limit_percent: int,
    result_queue: multiprocessing.Queue,
    input_folders: Optional[Sequence[str]] = None,
):
    """
    子进程工作函数：翻译一批图片

    ``input_folders``：输入的文件夹，输出时保留章节结构（不同章节的同名页不会写到同一路径）。
    设了内存上限时按 batch_size 分段翻译，段与段之间检查内存，超过上限就结束这个子进程，
    已完成的页照常回报，剩下的由下一个子进程继续（``restart``）。
    """
    import asyncio
    
    async def _do_translate():
        # 添加路径
        sys.path.insert(0, str(ROOT_DIR))
        sys.path.insert(0, str(ROOT_DIR / 'desktop_qt_ui'))
        
        import logging

        from manga_translator import Config, MangaTranslator
        from manga_translator.utils import init_logging, set_log_level
        
        init_logging()
        set_log_level(logging.DEBUG if verbose else logging.INFO)
        
        # 应用命令行参数
        cli_config = config_dict.get('cli', {})
        cli_config['verbose'] = verbose
        cli_config['overwrite'] = overwrite
        config_dict['cli'] = cli_config
        
        font_family = config_dict.get('render', {}).get('font_family')
        if font_family:
            config_dict['font_family'] = font_family
        # 创建翻译器
        translator_params = cli_config.copy()
        translator_params.update(config_dict)
        translator = MangaTranslator(params=translator_params)
        
        # 创建 Config 对象
        explicit_keys = {'render', 'upscale', 'translator', 'detector', 'colorizer', 'inpainter', 'ocr'}
        config_for_translate = {k: v for k, v in config_dict.items() if k in explicit_keys}
        for key in ['kernel_size', 'mask_dilation_offset', 'force_simple_sort']:
            if key in config_dict:
                config_for_translate[key] = config_dict[key]
        
        manga_config = Config(**config_for_translate)
        
        # 准备保存信息
        output_format = cli_config.get('format')
        if not output_format or output_format == "不指定":
            output_format = None
        
        save_info = {
            'output_folder': output_dir,
            'format': output_format,
            'overwrite': overwrite,
            'input_folders': {os.path.normpath(os.path.abspath(f)) for f in (input_folders or ())}
        }
        
        completed = []
        skipped = []
        failed = []
        items = [(file_path, manga_config) for file_path in file_paths]
        limited = memory_limit_mb > 0 or memory_limit_percent > 0
        step = max(1, int(cli_config.get('batch_size') or 1)) if limited else max(1, len(items))
        contexts = []
        for start in range(0, len(items), step):
            contexts.extend(await translator.translate_batch(items[start:start + step], save_info=save_info))
            over = over_memory_limit(memory_limit_mb, memory_limit_percent) if limited else None
            if over and start + step < len(items):
                print(f"📊 子进程内存超过上限（{over}）：已完成的页保存好了，换一个子进程继续剩下的 "
                      f"{len(items) - start - step} 页")
                restart[0] = True
                break

        for ctx in contexts:
            image_name = getattr(ctx, 'image_name', '') or ''
            file_name = os.path.basename(image_name) or '未知图片'
            if getattr(ctx, 'skipped', False):
                skipped.append(image_name)
                reason = getattr(ctx, 'skip_message', None) or '后端已跳过该文件'
                print(f"⏭️  跳过: {file_name} - {reason}")
            elif getattr(ctx, 'translation_error', None):
                failed.append(image_name)
                print(f"❌ 失败: {file_name} - {ctx.translation_error}")
            elif getattr(ctx, 'success', False) or getattr(ctx, 'result', None):
                completed.append(image_name)
                print(f"✅ 完成: {file_name}")
            else:
                failed.append(image_name)
                print(f"❌ 失败: {file_name} - 无返回结果")

        return completed, skipped, failed
    
    restart = [False]
    try:
        completed, skipped, failed = asyncio.run(_do_translate())
        result_queue.put({
            'status': 'success',
            'completed': completed,
            'skipped': skipped,
            'failed': failed,
            'restart': restart[0],
        })
    except Exception as e:
        import traceback
        print(f"\n❌ 子进程异常: {e}")
        result_queue.put({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'completed': [],
            'skipped': [],
            'failed': []
        })


async def translate_with_subprocess(
    all_files: List[str],
    output_dir: str,
    config_dict: dict,
    verbose: bool,
    overwrite: bool,
    memory_limit_mb: int = DEFAULT_MEMORY_THRESHOLD_MB,
    memory_limit_percent: int = DEFAULT_MEMORY_THRESHOLD_PERCENT,
    batch_per_restart: int = DEFAULT_BATCH_SIZE_PER_RESTART,
    resume: bool = False,
    input_folders: Optional[Sequence[str]] = None,
) -> Tuple[int, int, int]:
    """
    使用子进程模式翻译，支持内存管理
    
    Args:
        memory_limit_mb: 绝对内存限制（MB），0表示不限制
        memory_limit_percent: 内存百分比限制，超过系统总内存的这个百分比时重启
        input_folders: 输入的文件夹（保留章节结构）；会输出到同一路径的页不处理，算失败
    
    Returns:
        (success_count, skipped_count, failed_count)
    """
    completed_files = set()
    total_files = len(all_files)
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # 获取系统总内存用于显示
    total_mem = get_total_memory_mb()
    
    print(f"\n{'='*60}")
    print("🚀 子进程翻译模式")
    print(f"📊 总文件数: {total_files}")
    # 如果设置了绝对内存限制，只显示绝对限制；否则显示百分比限制
    if memory_limit_mb > 0:
        print(f"📊 内存限制: {memory_limit_mb} MB")
    elif memory_limit_percent > 0:
        limit_mb = total_mem * memory_limit_percent / 100
        print(f"📊 内存限制: {memory_limit_percent}% (约 {limit_mb:.0f} MB)")
    if batch_per_restart > 0:
        print(f"📊 每批处理: {batch_per_restart} 张")
    print(f"{'='*60}\n")
    
    output_format = (config_dict.get('cli') or {}).get('format')
    folders = [os.path.normpath(os.path.abspath(f)) for f in (input_folders or ())]
    clashes = same_output_paths(all_files, folders, output_format)
    for path, earlier in clashes.items():
        print(f"❌ 失败: {os.path.basename(path)} - 会和 {earlier} 输出到同一个路径，没有处理"
              f"（改用章节文件夹作为输入，或换输出格式）")
    failed_count += len(clashes)
    completed_files.update(clashes)
    
    restart_count = 0
    
    while True:
        # 每次循环开始时，过滤掉已完成的文件
        pending_files = [f for f in all_files if f not in completed_files]
        
        if not pending_files:
            break
        
        # 取一批文件处理（0 表示不限制，一次处理所有）
        if batch_per_restart > 0:
            batch_files = pending_files[:batch_per_restart]
        else:
            batch_files = pending_files
        
        print(f"\n{'='*40}")
        print(f"🔄 批次 {restart_count + 1}: 处理 {len(batch_files)} 个文件")
        print(f"📊 进度: {len(completed_files)}/{total_files}")
        print(f"{'='*40}")
        
        result_queue = multiprocessing.Queue()
        
        process = multiprocessing.Process(
            target=worker_translate_batch,
            args=(
                batch_files,
                output_dir,
                verbose,
                overwrite,
                config_dict,
                memory_limit_mb,
                memory_limit_percent,
                result_queue,
                folders,
            )
        )
        
        process.start()
        
        try:
            # 先尝试从队列获取结果（子进程会在发送结果后退出）
            timeout = len(batch_files) * 600
            try:
                result = result_queue.get(timeout=timeout)
                
                if result['status'] == 'success':
                    batch_completed = result.get('completed', [])
                    batch_skipped = result.get('skipped', [])
                    batch_failed = result.get('failed', [])

                    success_count += len(batch_completed)
                    skipped_count += len(batch_skipped)
                    failed_count += len(batch_failed)
                    completed_files.update(batch_completed)
                    completed_files.update(batch_skipped)
                    completed_files.update(batch_failed)

                    print(
                        f"\n📊 批次完成: 成功 {len(batch_completed)}, "
                        f"跳过 {len(batch_skipped)}, 失败 {len(batch_failed)}"
                    )
                    handled = set(batch_completed) | set(batch_skipped) | set(batch_failed)
                    if result.get('restart'):  # over the memory limit: the next worker goes on
                        print(f"🔄 内存超过上限，重启子进程继续剩下的 "
                              f"{sum(1 for f in batch_files if f not in handled)} 页")
                    elif not handled.issuperset(batch_files):  # pages the worker did not report
                        missing = [f for f in batch_files if f not in handled]
                        failed_count += len(missing)
                        completed_files.update(missing)
                else:
                    print(f"\n❌ 批次错误: {result.get('error', '未知错误')}")
                    if verbose and 'traceback' in result:
                        print(result['traceback'])
                    failed_count += len(batch_files)
                    completed_files.update(batch_files)
                    
            except Exception as e:
                print(f"\n⚠️ 无法获取子进程结果: {e}")
                # 如果无法获取结果，将这批文件标记为失败
                failed_count += len(batch_files)
                completed_files.update(batch_files)
            
            # 等待子进程退出
            process.join(timeout=30)
            if process.is_alive():
                print("⚠️ 子进程未正常退出，强制终止")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
        
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            raise
        
        main_mem = get_memory_usage_mb()
        if main_mem > 0:
            print(f"📊 主进程内存: {main_mem:.0f} MB")
        
        restart_count += 1
    
    if failed_count == 0:
        print("\n✅ 所有文件处理完成")
    else:
        print(f"\n⚠️ 有 {failed_count} 个文件失败")
    
    return success_count, skipped_count, failed_count
