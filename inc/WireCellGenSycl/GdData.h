#ifndef  GENSYCL_GDDATA_H
#define  GENSYCL_GDDATA_H    
namespace WireCell {
    namespace GenSycl {

       struct GdData {
           double p_ct ;
           double t_ct ;
	   double p_sigma ;
	   double t_sigma ;
	   double charge ;
           };

       struct DBin {
           double minval ;
           double binsize ;
	   int  nbins ;
       } ;

    }

}

#endif
