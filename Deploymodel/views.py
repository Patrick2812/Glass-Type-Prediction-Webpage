from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np



def home(request):
    return render(request,"home.html")

def result(request):

    cls = joblib.load('final_RFC_model.sav')

    lis = []

    lis.append(request.GET['RI'])
    lis.append(request.GET['Na'])
    lis.append(request.GET['Mg'])
    lis.append(request.GET['Al'])
    lis.append(request.GET['Si'])
    lis.append(request.GET['K'])
    lis.append(request.GET['Ca'])
    lis.append(request.GET['Ba'])
    lis.append(request.GET['Fe'])

    stdscaler = joblib.load('stdscaler.sav')
    lis1=list(map(float,lis))
    lis=np.asarray(lis1)
    lis=lis.reshape(1,-1)
    lis=stdscaler.transform(lis)
    ans = cls.predict(lis)


    return render(request,"result.html",{'ans':ans,'lis':lis1})