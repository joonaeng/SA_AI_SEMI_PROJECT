from flask import Flask
from flask import render_template
import cx_Oracle
from sqlalchemy import create_engine, text, Integer, Float, Numeric
import pandas as pd
import numpy as np


app = Flask(__name__)

@app.route('/')
def main111():

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9999)