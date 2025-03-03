from PIL import Image, ImageDraw, ImageFont, ImageOps

def draw_rotated_text(image, text, position, font_size, fill_color, angle, font_path='arial.ttf'):
    font = ImageFont.truetype(font_path, font_size)
    text_size = font.getbbox(text)[2:]  
    txt_img = Image.new('RGBA', text_size, (255, 255, 255, 0))
    txt_draw = ImageDraw.Draw(txt_img)
    txt_draw.text((0, 0), text, font=font, fill=fill_color, stroke_width=1, stroke_fill=(255, 255, 255, 0))
    rotated_txt = txt_img.rotate(angle, expand=True)
    image.paste(rotated_txt, position, rotated_txt)

def image_text():
    image = Image.open('resize.png')
    draw_rotated_text(image, 'dog', (100, 100), 20, (255, 255, 255, 255), 45)
    image.save('demo.png')

if __name__ == '__main__':
    image_text()