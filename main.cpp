
#include <iostream>
#include "memory.h"
#include "fstream"
#include "map"

#include "string"
#include "vector"

#include <cmath>


using namespace std;

//h(Q) = Q0 * 1 + Q1*x1 + Q2 * X2 + ...
//Qi (h(Q) - y) * x =  [Q0 + Q1 * X1 + Q2 * X2 + ... - y] * Xi
//Q0 (h(Q) - y) = (Q0 + Q1 * X1 + Q2 * X2 + ...- y) 


//Training samples 
map<int, pair<vector<double>, double>>samples;
//Training files
string trainfileName = "/Users/Cloud/Documents/dataminig/train_temp.csv";  

string testFileName = "/Users/Cloud/Documents/dataminig/test_temp.csv"; 
const int numOfFeatures = 384; 
double q[numOfFeatures + 1] = {0.0};
double a = 0.04;
int numC = 10000; 
size_t m = 1; 
const int lambda = 500;


// Used to read training samples
void readTrainSamples(string fileName){
    ifstream in(fileName);
    if (!in.is_open()){
        cout << "open file " << fileName << " failed" <<endl;
        exit(-1);
    }
    cout << "reading samples..." << endl;
    
    string feature;
    getline(in, feature); //去掉第一行的标签
    
    int id = 4;
    double value = 0; 
    double ref = 0; 
    char dot = ' '; 
    
    // Read the sample
    while(1){
        in >> id;
        if(in.eof())
            break;
        
        samples[id].first.clear(); 
        samples[id].first.push_back(1); // x0 = 1
  
        for (int i = 0; i < numOfFeatures; i++){
            in >> dot; 
            in >> value; 
            samples[id].first.push_back(value);
        }
        in >> dot;
        in >> ref; 
        samples[id].second = ref;
        while(in.peek() == '\n')
            in.get();
    }
    m = samples.size();
    in.close();
    
    cout << "read samples completed" << endl;
}


void training(){
    cout << "training ..." << endl;
    int j = 0;
    while(j < numC){
        double tmpQ[numOfFeatures+1] = {0.0};
        map<int, pair<vector<double>, double>>::iterator it = samples.begin();
        
        for ( ; it != samples.end(); it++){
            double b = 0.0; 
            for (int k = 0; k < it->second.first.size(); k++){
                b += (q[k] * it->second.first[k]);
            }
            b = b - it->second.second;
            tmpQ[0] += b;
            for (int i = 1; i < numOfFeatures + 1; i++){
                tmpQ[i] += (b * it->second.first[i]);
                
            }
        }
        tmpQ[0] = tmpQ[0] * a * (1.0/ m);
        q[0] = q[0] - tmpQ[0];
        for (int i = 1; i < numOfFeatures + 1; i++){
            tmpQ[i] = tmpQ[i] * a * (1.0/ m);
            q[i] = q[i] * (1 - a * lambda / m) - tmpQ[i];
        }
        j++;
        cout << "j = " << j << "\n";
        if (j >= 7000)
            a = 0.00001;
    }
    
}

void predict(){
    map<int, double> result;
    char t1[10];
    char t2[10];
    char t3[10];
    sprintf(t1, "%lf", a);
    sprintf(t2, "%d",numC);
    sprintf(t3, "%d", lambda);
    string fa = t1;
    string fc = t2;
    string fl = t3;
    
    ifstream in(testFileName);
    ofstream out("/Users/Cloud/Documents/dataminig/result_" + fa + "_" + fc + "_" + fl + ".csv");
    if (!in.is_open()){
        cout << "open file " << testFileName << " failed" <<endl;
        exit(-1);
    }
    if (!out.is_open()){
        cout << "open result file failed" << endl;
        exit(-1);
    }
    
    cout << "predicting ..." << endl;
    out << "Id,reference" << "\n";
    
    string feature;
    getline(in, feature); 
    
    int id = 0;
    char dot = ' ';
    double ref = 0.0;
    double value = -1; 
    

    while(1){
        in >> id; 
        if(in.eof())
            break;
        ref = 0.0;
        
        for (int i = 1; i < numOfFeatures + 1; i++){
            in >> dot; 
            in >> value;
            ref += (q[i] * value);
        }
        
        ref += q[0];
        out << id << "," << ref << "\n";
        while(in.peek() == '\n')
            in.get();
    }
    in.close();
    out.close();
    cout << "predict completed" << endl;
    
}


int main(int argc, const char * argv[]) {
    // insert code here...
    
    memset(q, 0, sizeof(q));
    readTrainSamples(trainfileName);
    training();
    predict();
    
    string qfile = "/Users/Cloud/Documents/dataminig/theta.csv";
    ofstream out(qfile);
    if (!out.is_open())
        exit(-1);
    
    for (int i = 0; i < numOfFeatures + 1; i++){
        out << i << " " << q[i] << "\n";
    }
    out.close();
    
    return 0;
}
