# -*- coding: utf-8 -*-

from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.shortcuts import render


import MultiClass_lstm.LstmMaster.code.Sentiment_lstm as sl
import MultiClass_lstm.LstmMaster.code.MutiClassify_lstm as MC



# 接收请求数据
def search(request):
    request.encoding = 'utf-8'
    line = request.GET['input']
    print("----------------------",line)
    if not line is None:
        message =sl.lstm_predict(line)
    else:
        message ='你提交了空表单'

    return HttpResponse(str(message))

def searchForNine(request):
    request.encoding = 'utf-8'
    line = request.GET['input']
    print(line)
    if 'input' in request.GET:
        message =MC.lstm_predict(line)
    else:
        message = '你提交了空表单'

    return HttpResponse(str(message))



# 表单
def search_form(request):
    return render_to_response('form.html')

def index(request):
    return render(request,"index.html",{"hello":"欢迎使用智能客服！"})