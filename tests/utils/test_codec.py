import numpy as np

from src.utils.codec import decode_base64_str_to_img, encode_img_to_base64_str


def test_decode_base64_str_to_img():
    # Given a base64 string for a blue pixel (png
    pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQIHWNgYPgPAAEDAQCarKlHAAAAAElFTkSuQmCC"
    # When we decode this string into an image
    pixel_img = decode_base64_str_to_img(pixel_base64)
    # Then we get the expected pixel
    assert pixel_img.shape == (1, 1, 3)
    assert np.array_equal(pixel_img, np.array([[[255, 0, 0]]]))


def test_encode_img_to_base64_str_default():
    # Given an image of a blue pixel
    pixel_img = np.array([[[255, 0, 0]]])
    # When we encode this image into a string
    pixel_b64 = encode_img_to_base64_str(pixel_img)
    # Then we get the expected base64 representation (jpg, much bigger than previous test due to jpg header)
    assert pixel_b64 == "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJB" \
                        "wYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCg" \
                        "oKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAE" \
                        "CAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2Jy" \
                        "ggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl" \
                        "5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAw" \
                        "EBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoE" \
                        "IFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3" \
                        "eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq" \
                        "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD8c6KKK/38Pys//9k="


def test_encode_img_to_base64_str_png():
    # Given an image of a blue pixel
    pixel_img = np.array([[[255, 0, 0]]])
    # When we encode this image into a string
    pixel_b64 = encode_img_to_base64_str(pixel_img, "png")
    # Then we get the expected base64 representation (jpg, much bigger than previous test due to jpg header)
    assert pixel_b64 == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQIHWNgYPgPAAEDAQCarKlHAAAAAElFTkSuQmCC"
