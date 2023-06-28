# Version 2.0
import camelot

def pdf_to_html_table(input_path, output_path):
    tables = camelot.read_pdf(input_path, flavor='stream', pages='all')

    html = "<html><body>"
    for table in tables:
        html += table.df.to_html(index=False, header=False)
    html += "</body></html>"

    with open(output_path, 'w') as file:
        file.write(html)

pdf_to_html_table('data.pdf', 'output.html')
