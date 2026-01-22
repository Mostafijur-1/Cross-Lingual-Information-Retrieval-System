import os, sys

# ensure module root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import app

def run_test():
    client = app.test_client()
    resp = client.get('/search?q=Bangladesh+cricket+victory&topk=3')
    print('status:', resp.status_code)
    data = resp.get_data(as_text=True)
    print('response snippet:\n', data[:1000])

if __name__ == '__main__':
    run_test()
