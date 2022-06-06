from PIL import Image
import pytesseract
import io
import re

NAME_ARABIC = "الاسم: "
NATIONALITY_ENGLISH = "Nationality: "


def read_id_img(img_data=None, img_path=None, text_data=None, lang="eng") -> str:
    """ Reading the id information from image. Returning the id number extracted. """
    # Added by eyad to fix the issue of tesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # tessdata_dir_config = '--tessdata-dir' + '"' + img_path + '"'


    assert img_data or img_path or text_data, "img_path or img_data or text_data parameters should be provided"
    if img_data:
        img = Image.open(io.BytesIO(img_data))
    elif img_path:
        img = Image.open(img_path)
    else:
        img = None

    if img:
        extract = pytesseract.image_to_string(img, config='--psm 1', lang=lang)
    elif text_data:
        extract = text_data
    else:
        return None

    extract = ''.join(extract)
    extract = [word for line in extract.split('\n') for word in line.split(" ") if len(word) == 11]

    if len(extract) == 0:
        return None

    for i in extract:
        xx = re.sub('\D', '', i)
        if len(xx) == 11:
            print('id: ' + xx)
            return xx
    return None


def parse_dob_str(dob_str:str) -> dict:
    """ Parsing DOB day, month, year from DOB str """

    dob_split = dob_str.split("/")
    if len(dob_split) != 3:
        return None

    data = {
        "d": int(dob_split[0]),
        "m": int(dob_split[1]),
        "y": int(dob_split[2])
    }
    return data


def read_dob_img(img_data=None, img_path=None, text_data=None) -> dict:
    """ Extracting D.O.B. from id image """

    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    # tessdata_dir_config = '--tessdata-dir' + '"' + img_path + '"'

    assert img_data or img_path or text_data, "img_path or img_data or text_data parameters should be provided"
    if img_data:
        img = Image.open(io.BytesIO(img_data))
    elif img_path:
        img = Image.open(img_path)
    else:
        img = None

    if img:
        extract = pytesseract.image_to_string(img, config='--psm 1')
    elif text_data:
        extract = text_data
    else:
        return None

    extractred_id = read_id_img(text_data=extract)
    if not extractred_id:
        return None
    extract = ''.join(extract)
    extract = [word for line in extract.split('\n') for word in line.split(" ") if len(word) == 10]
    if len(extract) == 0:
        return None

    for item in extract:
        if item[-2:] == extractred_id[1:3]:
            return parse_dob_str(item)
    return None


def read_name_img(img_data=None, img_path=None, text_data=None) -> str:
    """ Extracting the name from the id card scan """

    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    # tessdata_dir_config = '--tessdata-dir' + '"' + img_path + '"'

    assert img_data or img_path or text_data, "img_path or img_data or text_data parameters should be provided"
    if img_data:
        img = Image.open(io.BytesIO(img_data))
    elif img_path:
        img = Image.open(img_path)
    else:
        img = None

    if img:
        extract = pytesseract.image_to_string(img, config='--psm 1', lang='ara')
    elif text_data:
        extract = text_data
    else:
        return None

    extract = ''.join(extract)
    extract_lines = [line for line in extract.split('\n')]
    for line in extract_lines:
        if line.startswith(NAME_ARABIC):
            return line.replace(NAME_ARABIC, "")
    return None


def read_nationality_img(img_data=None, img_path=None, text_data=None) -> str:
    """ Extracting the name from the id card scan """

    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    # tessdata_dir_config = '--tessdata-dir' + '"' + img_path + '"'

    assert img_data or img_path or text_data, "img_path or img_data or text_data parameters should be provided"
    if img_data:
        img = Image.open(io.BytesIO(img_data))
    elif img_path:
        img = Image.open(img_path)
    else:
        img = None

    if img:
        extract = pytesseract.image_to_string(img, config='--psm 1', lang='eng')
    elif text_data:
        extract = text_data
    else:
        return None

    extract = ''.join(extract)
    extract_lines = [line for line in extract.split('\n')]
    for line in extract_lines:
        if line.startswith(NATIONALITY_ENGLISH):
            return line.replace(NATIONALITY_ENGLISH, "").lower()
    return None
