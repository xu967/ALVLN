#pragma once
// Minimal CPU-independent oracle for a single linear softmax CE layer.
// This is a verified numerical kernel, NOT the proposed TinyVLN student model.
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>
namespace tinyvln {
struct LinearCE {
  std::vector<double> w, x;
  int n_in, n_class;
  LinearCE(int n,int c):w(n*c,0),x(n,0),n_in(n),n_class(c) {}
  std::vector<double> logits() const {
    std::vector<double> out(n_class,0);
    for(int k=0;k<n_class;++k)for(int j=0;j<n_in;++j)
      out[k]+=w[k*n_in+j]*x[j];
    return out;
  }
  double loss(int label)const{
    if(label<0 ||label>=n_class)throw std::invalid_argument("LABEL_RANGE");
    auto v=logits();double maxv=*std::max_element(v.begin(),v.end());
    double z=0;for(auto a:v)z+=std::exp(a-maxv);
    return maxv+std::log(z)-v[label];
  }
  std::vector<double> grad(int label)const{
    auto v=logits();double maxv=*std::max_element(v.begin(),v.end());
    double z=0;for(auto a:v)z+=std::exp(a-maxv);
    std::vector<double> g(w.size(),0);
    for(int k=0;k<n_class;++k){
      double p=std::exp(v[k]-maxv)/z-(k==label?1.0:0.0);
      for(int j=0;j<n_in;++j)g[k*n_in+j]=p*x[j];
    }
    return g;
  }
};
}
