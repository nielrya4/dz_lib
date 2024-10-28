from io import BytesIO
from dz_lib.utils.formats import check
import base64

def buffer_to_utf8(buffer: BytesIO) -> str:
    buffer.seek(0)
    return buffer.getvalue().decode("utf-8")

def buffer_to_base64(buffer: BytesIO, img_format: str) -> str:
    buffer.seek(0)
    mime_type = f"image/{img_format}"
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_str}"

def fig_to_img_buffer(fig, fig_type="plotly", img_format="svg") -> BytesIO:
    accepted_image_formats = ['jpg', 'jpeg', 'png', 'pdf', 'eps', 'webp', 'svg']
    if fig_type == "plotly":
        if check(file_format=img_format, accepted_formats=accepted_image_formats):
            img_bytes = fig.to_image(format=img_format)
            buffer = BytesIO(img_bytes)
            return buffer
        else:
            raise ValueError(f"Unsupported format: {img_format}")
    elif fig_type == "matplotlib":
        if check(file_format=img_format, accepted_formats=accepted_image_formats):
            buffer = BytesIO()
            fig.savefig(buffer, format=img_format, bbox_inches="tight")
            buffer.seek(0)
            return buffer
    else:
        raise ValueError("fig_type must be either 'plotly' or 'matplotlib'")

def fig_to_html(fig, fig_type="plotly", vector=True) -> str:
    if fig_type == "plotly":
        return fig.to_html(full_html=False)
    elif fig_type == "matplotlib":
        if vector:
            img_format = "svg"
            img_buffer = fig_to_img_buffer(fig, fig_type=fig_type, img_format=img_format)
            html = f"<div>{buffer_to_utf8(img_buffer)}</div>"
        else:
            img_format = "png"
            img_buffer = fig_to_img_buffer(fig, fig_type=fig_type, img_format=img_format)
            html = f"<div><img src='{buffer_to_base64(img_buffer, img_format)}' /></div>"
    else:
        raise ValueError("fig_type must be either 'plotly' or 'matplotlib'")
    return html
