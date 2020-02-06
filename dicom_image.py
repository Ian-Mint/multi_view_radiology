from PIL import Image
import pydicom


def open(fp, mode="r"):
    """
    Designed to replicate the behavior of `PIL.Image.open`

    Args:
        fp: file path
        mode: unused. The only option is 'r'.

    Returns: a PIL image

    """
    dcm = pydicom.dcmread(fp)
    return Image.fromarray(dcm.pixel_array)
