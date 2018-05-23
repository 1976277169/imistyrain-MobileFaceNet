import sys
sys.path.insert(0,"D:\CNN\caffe\python")
import caffe

model="../models/MobileFaceNet_deploy.prototxt"

def main():
    net=caffe.Net(model,caffe.TEST)
    total=0
    for item in net.params.items():
        name,layer=item
        print(name+":"+str(layer[0].count))
        total+=layer[0].count
    print("total")
    print(total)
if __name__ == '__main__':
    main()