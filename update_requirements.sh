#!/bin/bash
pip install -r requirements-to-freeze.txt
pip freeze > requirements.txt
