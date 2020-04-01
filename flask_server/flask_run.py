from flask import Flask, request, Response, json
from flask_server.inference import get_response

app = Flask(__name__)


@app.route('/talk', methods=['POST', 'GET'])
def talk():
    content = request.args.get('content')
    result, _ = get_response(content)
    return Response(json.dumps({'nickName': '小AI', 'content': result, 'type': 'text'}), content_type='application/json')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8888, debug=False)
