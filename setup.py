from setuptools import setup, find_packages
setup(
    name="wastewise-backend",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'gunicorn', 
        'uvicorn',
        'flask',
        'flask-cors',
        'python-dotenv',
        'psycopg2-binary',
        'supabase',
        'postgrest'
    ],
)