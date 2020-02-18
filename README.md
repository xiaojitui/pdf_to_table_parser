# pdf_to_table_parser

The script is used to parse tables in PDFs to structured dataframes. 
<br><br>

PDF files should be put into 'samples' folder. 
To parse tables in the PDF, run:

`python pdf_to_table.py`
<br><br>

The script provides two methods of parsing tabels. 
* 'stream': detect table areas based on text edges, this is a algorithm implemented in the package called 'camelot'
'projection')
* 'projection': detect table areas based on element positions in each row. 

You can specify the method want to use in 'pdf_to_table.py'

<br><br>
The parsed tables can also be saved in the 'results' folder, using 'save_tables' function in 'pdf_to_table.py'

### Notes:
The script here is used to parse tables, not detect tables. 
Different algorithms of detecting table areas are provided at:
* 'get_pdf_row_features' repo
* 'object_detection' repo
* 'img_table_detector' repo
You can use above algorithms to detect table boundaries. 
<br><br>
If you already have table boundaries, you can specify the 'bbox' parameter in the 'pdf_to_table.py'. 
The format is:
* bbox = {'page_num': [[x1, y1, x2, y2]...]'} (e.g. bbox = {'0': [[100, 100, 600, 400]]})

If you don't have table boundaries, just leave 'bbox = None'. The script will use the default detection algorithms in 'camelot' package or 'pdfplumber' package to detect table areas in the PDF. 
