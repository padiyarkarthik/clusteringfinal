#importing libraries for csv,matplotlib,numpy,scipy and flask
import csv
from matplotlib import pylab
from pylab import *
import pylab
import numpy as np
import uuid
from numpy import vstack,array
from scipy.cluster.vq import *
from flask import Flask,render_template,request

app = Flask(__name__,template_folder="static")
coloumn_names = ["pclass","survived","name","sex","age","sibsp","parch","ticket","fare","cabin","embarked","boat","body","home.dest"]
myfile = open("titanic3.csv","r")
csv_reader = csv.DictReader(myfile, fieldnames=coloumn_names)
next(csv_reader)
#routing to index file
@app.route('/')
def index():
  return render_template('index.html')

mylist = []
@app.route('/kmeans', methods=['GET', 'POST'])
def main():
        att1 = request.form['attribute1']
        att2 = request.form['attribute2']
        clusters = request.form['clusters']
        K_clusters = int(clusters)
        mylist = getdata(att1,att2)
        data = []
        cdist=[]
        data = array(mylist)
        cent, pts = kmeans2(data,K_clusters)
        disCluster = []
        for i in range(len(cent)):
            x1 = cent[i][0]
            y1 = cent[i][1]
            x1 = float("{0:.3f}".format(x1))
            y1 = float("{0:.3f}".format(y1))

            for j in range(i+1,len(cent)):
                dc = {}
                x2 = cent[j][0]
                y2 = cent[j][1]
                x2 = float("{0:.3f}".format(x2))
                y2 = float("{0:.3f}".format(y2))
                dist = np.sqrt((x1-x2)*2 + (y1-y2)*2)
                cdist.append(dist)
                dc['dist'] = "Distance from cluster " + str(i) + " and cluster " + str(j) + " : " + str(dist)
                disCluster.append(dc)
                print (disCluster)
                print ("Distance from cluster " + str(i) + " and cluster " + str(j) + " : " + str(dist))
        clr = ([1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6,0.6,0.1],[1.0,0.5,0.0],[1.0,0.0,1.0],[0.6,0.2,0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00],[1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,	0.0, 1.0],[0.6,	0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00])
        rang = ([(clr)[i] for i in pts])
        rang_dict = {"yellow":0,"green":0,"red":0,"blue":0,"cyan":0}
        pdict=[]
        for x in rang:
            if str(x) == "[1, 1, 0.0]":
                rang_dict["yellow"] += 1
            if str(x) == "[0.2, 1, 0.2]":
                rang_dict["green"] += 1
            if str(x) == "[1, 0.2, 0.2]":
                rang_dict["red"] += 1
            if str(x) == "[0.3, 0.3, 1]":
                rang_dict["blue"] += 1
            if str(x) == "[0, 1.0, 1.0]":
                rang_dict["cyan"] += 1
            if str(x) == "[0.6, 0.6,0.1]":
                rang_dict["deepolive"] += 1
            if str(x) == "[1.0,	0.5, 0.0]":
                rang_dict["orange"] += 1
            if str(x) == "[1.0,	0.0, 1.0]":
                rang_dict["magenta"] += 1
            if str(x) == "[0.6,	0.2, 0.2]":
                rang_dict["ruby"] += 1
            if str(x) == "[0.1,	0.6, 0.6]":
                rang_dict["deepteal"] += 1
            if str(x) == "[0.0,	0.0, 0.0]":
                rang_dict["black"] += 1
            if str(x) == "[0.8,	1.0, 1.0]":
                rang_dict["palecyan"] += 1
            if str(x) == "[0.70, 0.50,	0.50]":
                rang_dict["dirtyviolet"] += 1
            if str(x) == "[0.5,	0.5, 0.5]":
                rang_dict["gray"] += 1
            if str(x) == "[0.77, 0.70, 0.00]":
                rang_dict["olive"] += 1

        f_write='Cluster,Count\r\n'
        cnt=0
        print (rang_dict)
        for i in rang_dict:
            if rang_dict[i] == 0:
                continue
            string = str(cnt) + " : " + str(rang_dict[i])
            pdict.append(string)
            print ("No of points in cluster with " + str(i) + " is: " + str(rang_dict[i]))
            f_write+= str(cnt)+','+str(rang_dict[i])+'\r\n'
            cnt += 1
        with open("static/d3chart.csv",'wb') as nfile:
            nfile.write(f_write.encode("utf-8"))
        pylab.scatter(data[:,0],data[:,1], c=rang)
        pylab.scatter(cent[:,0],cent[:,1], marker='o', s = 400, linewidths=3, c='none')
        pylab.scatter(cent[:,0],cent[:,1], marker='x', s = 400, linewidths=3)

        pylab.savefig("static/kmeans6.png")

        return render_template('index.html',cdist=cdist,pdict=pdict, disCluster = disCluster)

def getdata(attr1,attr2):
    c = 0
    for row in csv_reader:
        c += 1
        if c == 5000:
            break
        pair = []
        if row[attr1] == "":
            row[attr1] = 0
        if row[attr2] == "":
            row[attr2] = 0
        x = float(row[attr1])
        y = float(row[attr2])
        pair.append(x)
        pair.append(y)
        mylist.append(pair)
    return mylist

#routing to index file
@app.route('/show', methods=['GET', 'POST'])
def show():
  return render_template('show.html')
#routing to bargraph file
@app.route('/Bargraph', methods=['GET', 'POST'])
def bargraph():
  return render_template('d3barchart.html')
#routing to piegraph file
@app.route('/Piegraph', methods=['GET', 'POST'])
def Piegraph():
  return render_template('d3piechart.html')

#defining port

if __name__ == "__main__":
    app.run(debug=True,port=5010)