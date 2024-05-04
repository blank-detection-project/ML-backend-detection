# ML-backend-detection
Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Run FastApi backend
~~~
cd src
uvicorn backend:app --reload --port 8000   
~~~
Test API with Swagger:
http://127.0.0.1:8000/docs
