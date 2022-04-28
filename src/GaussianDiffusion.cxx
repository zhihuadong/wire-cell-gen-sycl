#include "WireCellGenSycl/GaussianDiffusion.h"

#include <iostream>		// debugging
//#include <omp.h>
#include <cstring>



#define MAX_NPSS_DEVICE 1000
#define MAX_NTSS_DEVICE 1000
#define FULL_MASK 0xffffffff
#define ZERO_THRESHOLD 1e-16



using namespace WireCell;
using namespace std;


double g_set_sampling_part1 = 0.0;
double g_set_sampling_part2 = 0.0;
double g_set_sampling_part3 = 0.0;
double g_set_sampling_part4 = 0.0;
double g_set_sampling_part5 = 0.0;



std::vector<double> GenSycl::GausDesc::sample(double start, double step, int nsamples) const
{
    std::vector<double> ret;
    
    if(!sigma){
        ret.resize(1, 0);
        ret[0] = 1;
        if(nsamples != 1){
            cerr<<"NOT one sample for true point source: "<<nsamples<<"\n";
        }
        
    }
    else{
        ret.resize(nsamples, 0.0);
        for (int ind=0; ind<nsamples; ++ind) {
            const double rel = (start + ind*step - center)/sigma;
            ret[ind] = exp(-0.5*rel*rel);
        }
    }

    return ret;
}

std::vector<double> GenSycl::GausDesc::binint(double start, double step, int nbins) const
{
    std::vector<double> bins;
    
    if(!sigma){
        bins.resize(1, 0);
        bins[0] = 1;
        if(nbins != 1){
            cerr<<"NOT one bin for true point source: "<<nbins<<"\n";
        }
    }
    else{
        bins.resize(nbins, 0.0);
        std::vector<double> erfs(nbins+1, 0.0);
        const double sqrt2 = sqrt(2.0);
        for (int ind=0; ind <= nbins; ++ind) {
            double x = (start + step * ind - center)/(sqrt2*sigma);
            erfs[ind] = 0.5*std::erf(x);
        }

        double tot = 0.0;
        for (int ibin=0; ibin < nbins; ++ibin) {
            const double val = erfs[ibin+1] - erfs[ibin];
            tot += val;
            bins[ibin] = val;
        }
    }
    return bins;
}

// integral Normal distribution with weighting function
// a linear weighting for the charge in each pbin
// Integral of charge spectrum <pvec> done by GausDesc::binint (do not do it again using Erf())
std::vector<double> GenSycl::GausDesc::weight(double start, double step, int nbins, std::vector<double> pvec) const
{
    std::vector<double> wt;
    if(!sigma){
        wt.resize(1, 0);
        wt[0] = (start+step - center)/step;
    }
    else{
        wt.resize(nbins, 0.0);
        const double pi = 4.0*atan(1);
        double x2 = start;
        double x1 = 0;
        double gaus2 = exp(-0.5*(start-center)/sigma*(start-center)/sigma);
        double gaus1 = 0;
        for (int ind=0; ind<nbins; ind++)
        {
            x1 = x2;
            x2 = x1 + step;
            double rel = (x2-center)/sigma;
            gaus1 = gaus2;
            gaus2 = exp(-0.5*rel*rel);


            // weighting
            wt[ind] = -1.0*sigma/(x1-x2)*(gaus2-gaus1)/sqrt(2.0*pi)/pvec[ind] + (center-x2)/(x1-x2);
            /* std::cerr<<"Gaus: "<<"1 and 2 "<<gaus1<<", "<<gaus2<<std::endl; */
            /* std::cerr<<"center, x1, x2: "<<center<<", "<<x1<<", "<<x2<<std::endl; */
            /* std::cerr<<"Total charge: "<<pvec[ind]<<std::endl; */
            /* std::cerr<<"weight: "<<ind<<" "<<wt[ind]<<std::endl; */
        }
    }
    return wt;
}



/// GaussianDiffusion

GenSycl::GaussianDiffusion::GaussianDiffusion(const IDepo::pointer& depo,
					  const GausDesc& time_desc, 
					  const GausDesc& pitch_desc)
    : m_deposition(depo)
    , m_time_desc(time_desc)
    , m_pitch_desc(pitch_desc)
    , m_toffset_bin(-1)
    , m_poffset_bin(-1)
{
}




void GenSycl::GaussianDiffusion::set_sampling(const Binning& tbin, // overall time tick binning
                                          const Binning& pbin, // overall impact position binning
                                          double nsigma,
                                          IRandom::pointer fluctuate,
                                          unsigned int weightstrat)
{
    if (m_patch.size() > 0) {
        return;
    }

    double wstart, wend;


    //wstart = omp_get_wtime();
    auto tval_range = m_time_desc.sigma_range(nsigma);
    auto tbin_range = tbin.sample_bin_range(tval_range.first, tval_range.second);
    const size_t ntss = tbin_range.second - tbin_range.first;
    m_toffset_bin = tbin_range.first;
    auto tvec =  m_time_desc.binint(tbin.edge(m_toffset_bin), tbin.binsize(), ntss);

    if (!ntss) {
        cerr << "Gen::GaussianDiffusion: no time bins for [" << tval_range.first/units::us << "," << tval_range.second/units::us << "] us\n";
        return;
    }

    auto pval_range = m_pitch_desc.sigma_range(nsigma);
    auto pbin_range = pbin.sample_bin_range(pval_range.first, pval_range.second);
    const size_t npss = pbin_range.second - pbin_range.first;
    m_poffset_bin = pbin_range.first;
    auto pvec = m_pitch_desc.binint(pbin.edge(m_poffset_bin), pbin.binsize(), npss);


    if (!npss) {
        cerr << "No impact bins [" << pval_range.first/units::mm << "," << pval_range.second/units::mm << "] mm\n";
        return;
    }


    if(weightstrat == 2){
        auto wvec = m_pitch_desc.weight(pbin.edge(m_poffset_bin), pbin.binsize(), npss, pvec);
        m_qweights = wvec;
    }
    if(weightstrat == 1){
        m_qweights.resize(npss, 0.5);
    }
    //wend = omp_get_wtime();
    g_set_sampling_part1 += wend - wstart;

   // wstart = omp_get_wtime();
    patch_t ret = patch_t::Zero(npss, ntss);
    double raw_sum=0.0;

    for (size_t ip = 0; ip < npss; ++ip) {
        for (size_t it = 0; it < ntss; ++it) {
            const double val = pvec[ip]*tvec[it];
            raw_sum += val;
            ret(ip,it) = (float)val;
        }
    }

    ret *= m_deposition->charge() / raw_sum;
    //wend = omp_get_wtime();
    g_set_sampling_part2 += wend - wstart;

    cout << "Before CPU ret(ip, it), charge=" << m_deposition->charge() << endl;
    for (size_t ip = 0; ip < npss; ++ip) {
        for (size_t it = 0; it < ntss; ++it) {
            cout << ret(ip,it) << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "set_sampling() : npss=" << npss << ", ntss=" << ntss << ", m_deposition->charge() = " << m_deposition->charge() << ", raw_sum=" << raw_sum << endl;


    //wstart = omp_get_wtime();
    const double charge_sign = m_deposition->charge() < 0 ? -1 : 1;

    double fluc_sum = 0;
    if (fluctuate) {
        double unfluc_sum = 0;

        for (size_t ip = 0; ip < npss; ++ip) {
          for (size_t it = 0; it < ntss; ++it) {
            const float oldval = ret(ip,it);
            unfluc_sum += oldval;

            float number = fluctuate->binomial((int)(std::abs(m_deposition->charge())), oldval/m_deposition->charge());
            fluc_sum += charge_sign*number;
            ret(ip,it) = charge_sign*number;
          }
        }
        if (fluc_sum == 0) {
           return;
        }
        else {
            ret *= m_deposition->charge() / fluc_sum;
        }
    }

    cout << "After CPU ret(ip, it)" << endl;
    for (size_t ip = 0; ip < npss; ++ip) {
        for (size_t it = 0; it < ntss; ++it) {
            cout << ret(ip,it) << " ";
        }
        cout << endl;
    }
    cout << endl;


    m_patch = ret;
    //wend = omp_get_wtime();
    g_set_sampling_part3 += wend - wstart;


}


void GenSycl::GaussianDiffusion::set_sampling_pre(
					        const int diff_idx ,
						int & patch_size ,
                                                double* p_vecs,
                                                double* t_vecs,
                                                double* charges, 
                                                unsigned long* p_idx,
                                                unsigned long * t_idx,
                                                unsigned long * patch_idx,
                                                const Binning& tbin, // overall time tick binning
                                                const Binning& pbin, // overall impact position binning
                                                double nsigma,
                                                IRandom::pointer fluctuate,
                                                unsigned int weightstrat)
{
    if (m_patch.size() > 0) {
        cerr << "GenSycl::GaussianDiffusion: patch size= 0 , diff number:" << diff_idx << "\n";
        p_idx[diff_idx+1] =p_idx[diff_idx] ;
        t_idx[diff_idx+1] =t_idx[diff_idx] ;
        patch_idx[diff_idx+1] =patch_idx[diff_idx] ;
        return;
    }

    double wstart=0.0, wend=0.0;


    //wstart = omp_get_wtime();
    /// Sample time dimension
    auto tval_range = m_time_desc.sigma_range(nsigma);
    auto tbin_range = tbin.sample_bin_range(tval_range.first, tval_range.second);
    const size_t ntss = tbin_range.second - tbin_range.first;
    m_toffset_bin = tbin_range.first;
    auto tvec =  m_time_desc.binint(tbin.edge(m_toffset_bin), tbin.binsize(), ntss);

    if (!ntss) {
        cerr << "GenSycl::GaussianDiffusion: no time bins for [" << tval_range.first/units::us << "," << tval_range.second/units::us << "] us\n";
        p_idx[diff_idx+1] =p_idx[diff_idx] ;
        t_idx[diff_idx+1] =t_idx[diff_idx] ;
        patch_idx[diff_idx+1] =patch_idx[diff_idx] ;
        return;
    }

    /// Sample pitch dimension.
    auto pval_range = m_pitch_desc.sigma_range(nsigma);
    auto pbin_range = pbin.sample_bin_range(pval_range.first, pval_range.second);
    const size_t npss = pbin_range.second - pbin_range.first;
    m_poffset_bin = pbin_range.first;
    auto pvec = m_pitch_desc.binint(pbin.edge(m_poffset_bin), pbin.binsize(), npss);
    

    if (!npss) {
        cerr << "No impact bins [" << pval_range.first/units::mm << "," << pval_range.second/units::mm << "] mm\n";
        p_idx[diff_idx+1] =p_idx[diff_idx] ;
        t_idx[diff_idx+1] =t_idx[diff_idx] ;
        patch_idx[diff_idx+1] =patch_idx[diff_idx] ;
        return;
    }

    patch_size = ntss*npss ;

    // make charge weights for later interpolation.
    /// fixme: for hanyu.
    if(weightstrat == 2){
        auto wvec = m_pitch_desc.weight(pbin.edge(m_poffset_bin), pbin.binsize(), npss, pvec);
        m_qweights = wvec;
    }
    if(weightstrat == 1){
        m_qweights.resize(npss, 0.5);
    }
    //wend = omp_get_wtime();
    g_set_sampling_part1 += wend - wstart;

    m_patch = patch_t::Zero(npss, ntss);


    //wstart = omp_get_wtime();
   
    memcpy(&p_vecs[p_idx[diff_idx]], &pvec[0], npss * sizeof(double));
    memcpy(&t_vecs[t_idx[diff_idx]], &tvec[0], ntss * sizeof(double));
    charges[diff_idx]=m_deposition->charge() ;
    p_idx[diff_idx+1] =npss+p_idx[diff_idx] ;
    t_idx[diff_idx+1] =ntss+t_idx[diff_idx] ;
    patch_idx[diff_idx+1] =npss*ntss+patch_idx[diff_idx] ;

    //wend = omp_get_wtime();
    g_set_sampling_part2 += wend - wstart;
}

void GenSycl::GaussianDiffusion::clear_sampling(){
  m_patch.resize(0,0); 
  m_qweights.clear();
  m_qweights.shrink_to_fit();
}

// patch = nimpacts rows X nticks columns
// patch(row,col)
const GenSycl::GaussianDiffusion::patch_t& GenSycl::GaussianDiffusion::patch() const
{
    return m_patch;
}

const std::vector<double> GenSycl::GaussianDiffusion::weights() const
{
    return m_qweights;
}


