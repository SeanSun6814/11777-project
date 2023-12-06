from flask import Flask, request, jsonify
from transition_amr_parser.parse import AMRParser

app = Flask(__name__)

# Initialize the parser outside of the request handling to avoid reloading it on each request
parser = AMRParser.from_pretrained('AMR3-structbart-L')

@app.route('/parse', methods=['POST'])
def parse_amr():
    data = request.json
    sentence = data.get('sentence')

    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    try:
        tokens, positions = parser.tokenize(sentence)
        annotations, machines = parser.parse_sentence(tokens)
        amr = machines.get_amr()
        amr_str = amr.to_penman(jamr=False, isi=True)
        return jsonify({'amr': amr_str})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the application on all available IPs and on port 8080 (you can choose any port)
    app.run(host='0.0.0.0', port=8080, debug=True)
