import numpy as np
from flask import Flask, request, jsonify

from player import AIServer

app = Flask(__name__)
ai = None


def setup():
    global ai
    ai = AIServer((9, 9), model_id=310, iteration=1600)


with app.app_context():
    setup()


@app.route('/make_move', methods=['POST'])
def make_move():
    global ai
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    try:
        state = np.array(data['state'], dtype=np.float32)
        last_action = int(data['last_action'])

    except Exception as e:
        return jsonify({"error": f'Failed to parse input:{e}'}), 400

    ai.run_mcts(state, last_action)

    return jsonify({"action": int(ai.pending_action)})


@app.route('/reset', methods=['GET'])
def reset():
    global ai
    if ai:
        ai.reset()
    return jsonify({"status": 'success'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
