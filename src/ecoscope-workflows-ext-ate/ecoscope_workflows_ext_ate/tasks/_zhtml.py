from typing import Annotated, Union
from pathlib import Path
from multiprocessing import Pool

from ecoscope_workflows_core.decorators import task
from pydantic import BaseModel, Field



class ScreenshotConfig(BaseModel):
    width: int = 1280
    height: int = 720
    full_page: bool = False
    device_scale_factor: float = 2.0
    wait_for_timeout: int = 60_000


def _convert_html_to_png(
    html_path: str,
    output_dir: str,
    config: ScreenshotConfig,
) -> str:
    from playwright.sync_api import sync_playwright
    """Helper function with the core conversion logic."""
    png_filename = Path(html_path).with_suffix(".png").name

    if output_dir.startswith("file://"):
        output_dir = output_dir[7:]

    output_path = Path(output_dir) / png_filename

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": config.width, "height": config.height},
            device_scale_factor=config.device_scale_factor,
        )
        page.goto(Path(html_path).as_uri())
        page.wait_for_load_state("networkidle", timeout=0)
        page.wait_for_timeout(config.wait_for_timeout)
        page.screenshot(path=output_path, full_page=True, timeout=0)
        browser.close()
    return str(output_path)


def _html_to_png_worker(args):
    """Worker function for the multiprocessing pool."""
    html_path, output_dir, config = args
    return _convert_html_to_png(html_path, output_dir, config)


@task
def zhtml_to_png(
    html_path: Annotated[Union[str, list[str]], Field(description="The html file path(s)")],
    output_dir: Annotated[str, Field(description="The output root path")],
    config: Annotated[
        ScreenshotConfig, Field(description="The screenshot configuration")
    ] = ScreenshotConfig(),
) -> Union[str, list[str]]:
    """
    Task to convert a single HTML file or a list of HTML files to PNG images.
    If a list is provided, the conversion is done in parallel using multiprocessing.

    To speed up the process for multiple files, provide a list of html_path.
    This will use a multiprocessing pool to distribute the work across multiple CPU cores,
    significantly reducing the total processing time.
    """
    if isinstance(html_path, str):
        return _convert_html_to_png(html_path, output_dir, config)

    # list of paths, use multiprocessing
    args_list = [(path, output_dir, config) for path in html_path]
    with Pool() as pool:
        output_paths = pool.map(_html_to_png_worker, args_list)
    return output_paths