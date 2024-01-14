import os

import requests
import vertexai
from flask import Flask, jsonify, render_template, request
from vertexai.preview.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)

global chat_model

FAKE_ORDER_RESPONSE = [{'orderId': '12345678', "status": 'shipped', 'zipcode': '1624GC'},
                       {'orderId': '22233377', 'status': 'picking', 'zipcode': '3333AA'}]

app = Flask(__name__)
PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)

get_order_info_func = FunctionDeclaration(
    name="get_order_status",
    description="Get information about an order",
    parameters={
        "type": "object",
        "properties": {
            "orderid": {"type": "string", "description": "The order ID"},
            "zipcode": {"type": "string", "description": "The zipcode where the order was shipped to"},
        },
        "required": [
            "orderid",
            "zipcode"
        ]
    },
)

retail_functions = Tool(
    function_declarations=[
        get_order_info_func
    ],
)


def create_session():
    chat_model = GenerativeModel(
        "gemini-pro", generation_config={"temperature": 0}, tools=[retail_functions]
    )
    chat = chat_model.start_chat()
    return chat


def do_function_call(chat, function_name, params):
    url = "http://localhost:8080/orders?"
    for param in params:
        url += '{}={}&'.format(param, params[param])
    url += "format=json"

    api_response = requests.get(url)
    json_response = api_response.json()

    # call vertex ai for a nice response
    nice_response = chat.send_message(
        Part.from_function_response(
            name="get_order_status",
            response={
                "content": json_response,
            },
        ),
    )

    # show response to the user
    nice_text = nice_response.candidates[0].content.parts[0]
    return nice_text.text


def is_function_call(part):
    return part.function_call.args is not None


def response(chat, message):
    result = chat.send_message(message)
    # get the response parameters from the function response
    if is_function_call(result.candidates[0].content.parts[0]):
        params = result.candidates[0].content.parts[0].function_call.args
        return do_function_call(chat, result.candidates[0].content.parts[0].function_call.name, params)
    # get the api response
    return result.text


@app.route("/")
def index():
    ###
    return render_template("index.html")


@app.route("/chat", methods=["GET", "POST"])
def vertex_chat():
    user_input = ""
    if request.method == "GET":
        user_input = request.args.get("user_input")
    else:
        user_input = request.form["user_input"]

    content = response(chat_model, user_input)
    return jsonify(content=content)


@app.route("/orders")
def orders():
    for order in FAKE_ORDER_RESPONSE:
        if order["orderId"] == request.args.get("orderid") and order["zipcode"] == request.args.get("zipcode"):
            return jsonify(order)

    return jsonify({'error': 'Order not found'})


if __name__ == "__main__":
    chat_model = create_session()
    app.run(debug=True, port=8080, host="0.0.0.0")
