import http.client
import json
import sys
sys.path.append(' ')
from account import ACCOUNT
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

def aliyun_recong(audio_path):
    appKey =  ACCOUNT["Alibaba"]["app_key"]
    access_key_id=ACCOUNT["Alibaba"]["access_key_id"]
    access_key_secret=ACCOUNT["Alibaba"]["access_key_secret"]

    client = AcsClient(access_key_id, access_key_secret, "cn-shanghai")
    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('CreateToken')
    response = client.do_action_with_exception(request)

    token = str(response, 'utf-8')
    token = json.loads(token)
    token = token['Token']['Id']

    url = 'http://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr'

    audioFile = audio_path
    format = 'wav'
    sampleRate = 16000
    enablePunctuationPrediction  = True
    enableInverseTextNormalization = True
    enableVoiceDetection  = False

    request = url + '?appkey=' + appKey
    request = request + '&format=' + format
    request = request + '&sample_rate=' + str(sampleRate)

    if enablePunctuationPrediction :
        request = request + '&enable_punctuation_prediction=' + 'true'

    if enableInverseTextNormalization :
        request = request + '&enable_inverse_text_normalization=' + 'true'

    if enableVoiceDetection :
        request = request + '&enable_voice_detection=' + 'true'

    # print('Request: ' + request)

    asr_result = process(request, token, audioFile)

    return asr_result


def process(request, token, audioFile) :
    with open(audioFile, mode = 'rb') as f:
        audioContent = f.read()

    host = 'nls-gateway.cn-shanghai.aliyuncs.com'
    
    httpHeaders = {
        'X-NLS-Token': token,
        'Content-type': 'application/octet-stream',
        'Content-Length': len(audioContent)
        }

    # Python 2.x->httplib
    # conn = httplib.HTTPConnection(host)

    # Python 3.x->http.client
    conn = http.client.HTTPConnection(host)

    conn.request(method='POST', url=request, body=audioContent, headers=httpHeaders)

    response = conn.getresponse()
    # print('Response status and response reason:')
    # print(response.status ,response.reason)

    body = response.read()
    success_flag=False
    try:
        # print('Recognize response is:')
        body = json.loads(body)
        # print(body)

        status = body['status']
        if status == 20000000 :
            result = body['result']
            success_flag=True
            # print('Recognize result: ' + result)
        else :
            result = 'Recognizer failed!'

    except ValueError:
        result = "The response is not json format string"

    conn.close()

    return result,success_flag
