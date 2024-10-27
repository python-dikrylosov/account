from flask import Flask, request, render_template, send_file
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from werkzeug.utils import secure_filename

# Создаем экземпляр приложения Flask
app = Flask(__name__)

# Указываем папку для хранения загружаемых файлов
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ограничиваем типы файлов, которые могут быть загружены
ALLOWED_EXTENSIONS = {'xls', 'xlsx', 'docx', 'pdf'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Проверяем наличие файла в запросе
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # Если пользователь не выбрал файл
        if file.filename == '':
            return 'Пожалуйста, выберите файл для загрузки.'

        # Извлечение имени файла
        filename = secure_filename(file.filename)

        if not allowed_file(file.filename):
            return f'Файл {filename} имеет недопустимый формат. Разрешены форматы: .xls, .xlsx, .docx, .pdf'

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        df = None
        extension = filename.split('.')[-1]

        try:
            if extension in ['xls', 'xlsx']:
                df = pd.read_excel(f'{UPLOAD_FOLDER}/{filename}')
            elif extension == 'docx':
                df = docx_to_dataframe(f'{UPLOAD_FOLDER}/{filename}')
            else:
                df = pdf_to_dataframe(f'{UPLOAD_FOLDER}/{filename}')

            fig = create_plot(df)
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            return send_file(output, mimetype='image/png', attachment_filename=f'plot.png', as_attachment=True)
        except Exception as e:
            print(e)
            return str(e)

    return '''
    <!doctype html>
    <html lang="ru">
      <head>
        <!-- Required meta tags -->
        <meta charset="utf-8">
        <title>Загрузка файла</title>
      </head>
      <body>
        <h1>Загрузите файл</h1>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="file">
          <button type="submit">Отправить</button>
        </form>
      </body>
    </html>
    '''


def create_plot(dataframe):
    fig, ax = plt.subplots()
    dataframe.plot(ax=ax)
    return fig


def docx_to_dataframe(docx_path):
    # Импортируем библиотеки для работы с docx
    from docx import Document
    document = Document(docx_path)

    data = []
    for table in document.tables:
        rows = []
        keys = None
        for i, row in enumerate(table.rows):
            text = (cell.text for cell in row.cells)
            if i == 0:
                keys = tuple(text)
                continue
            row_data = dict(zip(keys, text))
            rows.append(row_data)
        df = pd.DataFrame(rows)
        data.append(df)
    final_df = pd.concat(data)
    return final_df


def pdf_to_dataframe(pdf_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    text = page.extract_text()
    lines = text.split('\n')
    header = lines[0].strip().split(',')
    rows = [line.strip().split(',') for line in lines[1:] if line.strip()]
    df = pd.DataFrame(rows, columns=header)
    df.dropna(how='all', inplace=True)
    df.replace('', float('nan'), inplace=True)
    df.fillna(method='ffill', axis=1, inplace=True)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return df


if __name__ == '__main__':
    app.run(debug=True)
