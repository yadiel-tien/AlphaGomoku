import numpy as np
from flask import Flask, request, jsonify

from inference import InferenceEngine as Engine
from player import AIServer

app = Flask(__name__)
AIes: list[AIServer | None] = [None, None]


@app.route('/setup', methods=['POST'])
def setup():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400
    try:
        player_idx = int(data['player_idx'])
        model_idx = int(data['model_idx'])
        env_name = data['env_class']
        infer = Engine.make_engine(model_idx)
        AIes[player_idx] = AIServer(infer, env_name, 1000)
    except Exception as e:
        print(f'Failed to setup AI: {e}')
        return jsonify({"error": f'Failed to setup AI:{e}'}), 400

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
        player_to_move = int(data['player_to_move'])
        action = AIes[player_idx].get_action(state, last_action, player_to_move)
        return jsonify({"last_action": action})
    except Exception as e:
        return jsonify({"error": f'Failed to make move:{e}'}), 400


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
        return jsonify({"error": f'Failed to reset:{e}'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
