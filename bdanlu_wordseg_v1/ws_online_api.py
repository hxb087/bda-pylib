# -*- coding: utf-8 -*-
# ==============================
# @Time   :   2020/5/21 11:25
# @Auth   :   zhou
# @File   :   online_api
# ==============================

import json
from flask import Flask, request, jsonify
from flasgger import Swagger, NO_SANITIZER
from flasgger.utils import swag_from
from flasgger import LazyString, LazyJSONEncoder
from bilstm_crf import call_bilstm_crf

segmenter = call_bilstm_crf()

app = Flask(__name__)
app.config["SWAGGER"] = {"title": "Golaxy文本分词标注抽取API", "uiversion": 2}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "/wss/v1_ws",
            "route": "/wss/v1/ws.json",
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/wss/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/wss/index.html",
}

template = {
  "swagger": "2.0",
  "info": {
    "title": "Golaxy文本分词标注抽取API",
    "description": "Golaxy文本分词标注抽取API",
    "contact": {
      "responsibleOrganization": "zhousl",
      "responsibleDeveloper": "zhousl",
      "email": "zhoushaolong@golaxy.cn",
      "url": "www.golaxy.cn",
    },
    "termsOfService": "zhoushaolong@golaxy.cn",
    "version": "2.0.0"
  },
  "produces": "application/json",
  "consumes": "application/json",
  # "host": "127.0.0.1:8011",  # overrides localhost:500
  # "basePath": "/golaxy/nlp/wss/",  # base bash for blueprint registration
  "schemes": [
    "http",
    "https"
  ],
  "swaggerUiPrefix": LazyString(lambda: request.environ.get("HTTP_X_SCRIPT_NAME", ""))
}

app.json_encoder = LazyJSONEncoder
swagger = Swagger(app, config=swagger_config, template=template) #, sanitizer=NO_SANITIZER)


def get_zh_predict(sentence):
    data = []
    data.append(sentence)
    output = segmenter.decode_text(data)[0]
    return output

@app.route("/wss/chinese/v1", methods=["POST"])
@swag_from("ws_swagger_zh.yml")
def get_zh_ws():
    input_json = request.get_json()
    resout = {}
    try:
        str1 = str(input_json["text"])
        res = get_zh_predict(str1)
        resout['message'] = "success"
        resout['status'] = 0
        resout['ws'] = res
        resout['total'] = len(res)
    except Exception as e:
        print(e)
        resout['message'] = "success"
        resout['status'] = 1
        resout['ws'] = []
        resout['total'] = 0
    return json.dumps(resout, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8021', debug=True)