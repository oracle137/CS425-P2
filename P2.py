import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def dunn(loc,pcdata):

    max_diameter = diameter(loc,pcdata)
    min_dist = min_cluster_distances(loc,pcdata)
    return min_dist/max_diameter,max_diameter,min_dist


def min_cluster_distances(loc,pcdata):
    min_distances = -1
    for i in range(0,len(loc)):
        for j in range(i+1,len(loc)):
            for ivalue in loc[i]:
                for jvalue in loc[j]:
                    dist = np.linalg.norm(pcdata[ivalue[1]] - pcdata[jvalue[1]])
                    if min_distances == -1:
                        min_distances = dist
                    elif dist < min_distances:
                        min_distances = dist
    return min_distances


def diameter(loc,pcdata):
    max = 0
    for i in range(0,len(loc)):
        for j in range(0,len(loc[i])):
            for k in range(j+1,len(loc[i])):
                dist = np.linalg.norm(pcdata[loc[i][j][1]] - pcdata[loc[i][k][1]])
                if dist > max:
                    max  = dist
    return max


def plotgraphs(vals,vals2,plotname,xlabel,ylabel):
    plt.plot(vals,'o-')
    plt.title(plotname)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    leg = plt.legend([plotname], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()


def plotscatter(vals,vals2,plotname):
    # fig = plt.figure(figsize=(8, 5))

    plt.scatter(vals,vals2)
    plt.title(plotname)
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    leg = plt.legend([plotname], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()


def resetCluster(k):
    loc = []
    for j in range(0, k):
        loc.append([])
    return loc


def checkChange(prevcluster,newcluster):
    booll = False
    for i in range(0,len(prevcluster)):
        if type(newcluster[i]) != np.float64:
            if (set(prevcluster[i][0]) != set(newcluster[i][0])):
                booll = True
        else:
            newcluster[i] = prevcluster[i-1]
    return booll


def getClusters(values,k):
    clusters = []
    for i in range(0,k):
        random_indexx = np.random.randint(0, values.shape[0])
        clusters.append([values.A[random_indexx],random_indexx])
    return clusters


def clusterFinding(values,pcdata,k,title):
    clusters = getClusters(values,k)
    prevCluster = getClusters(values, k)
    loc = resetCluster(k)
    # plt.ion()
    fig, ax = plt.subplots()
    iterations = 0
    while checkChange(prevCluster,clusters):
        loc = resetCluster(k)
        for i in range(0,values.shape[0]):
            loc[min(range(0,len(clusters)), key=lambda index: np.linalg.norm(clusters[index][0] - values.A[i]))].append([values.A[i],i])
        prevCluster = clusters.copy()
        clusters.clear()
        for i in loc:
            clusters.append(np.mean(i,0))
        iterations += 1

    colors = ["g", "c", "b", "k", "m", "y", "w"]
    for i in range(0,len(clusters)):
        for j in range(0,len(loc[i])):
            plt.scatter(pcdata[loc[i][j][1]][0],pcdata[loc[i][j][1]][1],color=colors[i])
            # fig.canvas.draw()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.show()
    print("Convergence takes: " + str(iterations))
    x, y, z = dunn(loc,pcdata)
    print(str(round(x,3)),str(round(y,2)),str(round(z,2)))


def start():
    df = pd.read_csv('under5mortalityper1000.csv',sep=',',usecols=range(1,217)).dropna(how='any')
    df = df.fillna(df.mean())

    np.random.seed(5002)

    data = np.matrix(df)
    np.random.shuffle(data)
    print(data)
    u,s,v = np.linalg.svd(data,full_matrices=True)
    [plt.plot(range(0,len(df.values[i])),df.values[i],'.') for i in range(0, len(df.values))]
    plt.xlabel("Years")
    plt.ylabel("Deaths/1000")

    plt.show()

    plotgraphs(s,np.arange(len(s)) + 1,'Value','Principal Components','Singular Value')


    s = s**2
    ksum = 0
    for sum in s:
        ksum = ksum + sum
    variancefactor = np.cumsum(s) / ksum
    plotgraphs(variancefactor,np.arange(len(variancefactor)) + 1,'Percent of Variance covered by the first k Singular values','Principal Compnents','Percent')

    k = 3
    first14PC = v[:,:k]
    newdata = np.flip(np.transpose(first14PC)*np.transpose(data),axis=0)

    A = newdata[0,:].A[0]
    B = newdata[1,:].A[0]
    plotscatter(np.ndarray.tolist(newdata[0,:]),np.ndarray.tolist(newdata[1,:]),'Scatter Plot of First two PC')

    pcdata = np.column_stack((A,B))
    newdataT = np.transpose(newdata)

    clusterFinding(data,pcdata,k,'Full Data Matrix Input')
    clusterFinding(newdataT,pcdata,k,'PC Matrix Input')
    clusterFinding(np.matrix(pcdata),pcdata,k,'First Two PCs')


start()