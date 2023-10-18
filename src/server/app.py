from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # TODO: Here, you'll later add the code to process the image through your ML model.
    # For now, we'll just return a placeholder message.
    return jsonify({'message': 'File received!'})

if __name__ == '__main__':
    app.run(debug=True)  # Run the server in debug mode for development
