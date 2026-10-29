#include "tensor_ops.hpp"
#include <cmath>
#include <iostream>
int main(){
  tinyvln::LinearCE m(4,3);
  m.x={0.7,-0.2,0.3,0.5};
  for(int i=0;i<12;++i)m.w[i]=.12*(i-5);
  const double eps=1e-6;auto exact=m.grad(2);double max_err=0;
  for(size_t i=0;i<m.w.size();++i){
    double orig=m.w[i];m.w[i]=orig+eps;double pos=m.loss(2);
    m.w[i]=orig-eps;double neg=m.loss(2);m.w[i]=orig;
    max_err=std::max(max_err,std::abs((pos-neg)/(2*eps)-exact[i]));
  }
  std::cout<<"TinyVLN CPU linear/CE FP64 gradcheck max_abs="<<max_err<<"\n";
  if(max_err>1e-7)return 1;
  std::cout<<"[PASS][NUMERIC_GRADCHECK] Not a drone training result.\n";
  return 0;
}
