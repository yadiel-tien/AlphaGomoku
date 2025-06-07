import numpy as np
import torch
from flask import Flask, request, jsonify

from inference import make_engine
from player import AIServer

app = Flask(__name__)
AIes = [None, None]


@app.route('/setup', methods=['POST'])
def setup():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400
    try:
        player_idx = int(data['player_idx'])
        model_idx = int(data['model_idx'])
        infer = make_engine(model_idx)
        AIes[player_idx] = AIServer(infer, 1000)
    except Exception as e:
        print(f'Failed to setup AI: {e}')
        return jsonify({"error": f'Failed to parse input:{e}'}), 400

    return jsonify({"status": "success"})


@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    try:
        state = np.array(data['state'], dtype=np.float32)
        last_action = data['last_action']
        if last_action is not None:
            last_action = int(last_action)
        player_idx = int(data['player_idx'])

        AIes[player_idx].run_mcts(state, last_action)
        action = int(AIes[player_idx].pending_action)
        return jsonify({"action": action})
    except Exception as e:
        return jsonify({"error": f'Failed to parse input:{e}'}), 400


@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    try:
        player_idx = int(data['player_idx'])
        if AIes[player_idx] is not None:
            AIes[player_idx].reset()
            return jsonify({"status": 'success'})
        else:
            return jsonify({"error": "Player have not been setup"}), 400
    except Exception as e:
        return jsonify({"error": f'Failed to parse input:{e}'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
