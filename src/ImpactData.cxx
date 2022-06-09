#include "WireCellGenSycl/ImpactData.h"

#include <iostream>             // debugging

using namespace WireCell;
using namespace std;

GenSycl::ImpactData::ImpactData(int impact)
    : m_impact(impact)
{
}
void GenSycl::ImpactData::add(GaussianDiffusion::pointer diffusion)
{
    m_diffusions.push_back(diffusion);
}

Waveform::realseq_t& GenSycl::ImpactData::waveform() const
{
    return m_waveform;
}

Waveform::compseq_t& GenSycl::ImpactData::spectrum() const
{
    return m_spectrum;
}

Waveform::realseq_t& GenSycl::ImpactData::weightform() const
{
    return m_weights;
}

Waveform::compseq_t& GenSycl::ImpactData::weight_spectrum() const
{
    return m_weight_spectrum;
}

void GenSycl::ImpactData::calculate(int nticks) const
{
    if (m_waveform.size() > 0) {
        return;
    }
    m_waveform.resize(nticks, 0.0);
    m_weights.resize(nticks, 0.0);

    for (auto diff : m_diffusions) {

	const auto patch = diff->patch();
	const auto qweight = diff->weights();

        const int poffset_bin = diff->poffset_bin();
        const int pbin = m_impact - poffset_bin;
        const int np = patch.rows();
        if (pbin<0 || pbin >= np) {
            continue;
        }

        const int toffset_bin = diff->toffset_bin();
        const int nt = patch.cols();

	//	std::cout << pbin << " " << poffset_bin << " " << m_impact << std::endl;
	
        for (int tbin=0; tbin<nt; ++tbin) {
            const int absbin = tbin+toffset_bin;
            m_waveform[absbin] += patch(pbin, tbin);

	    //  std::cout << pbin << " " << tbin << " " << patch(pbin,tbin) << std::endl;
	    
            // for interpolation
            m_weights[absbin] += qweight[pbin]*patch(pbin, tbin);
        }
    }

    m_spectrum = Waveform::dft(m_waveform);
    m_weight_spectrum = Waveform::dft(m_weights);
}


// std::pair<int,int> GenSycl::ImpactData::strip() const
// {
//     int imin=-1, imax = -1;
//     for (int ind=0; ind<m_waveform.size(); ++ind) {
//         const double val = m_waveform[ind];
//         if (imin < 0 && val > 0) { imin = ind; }
//         if (val > 0) { imax = ind; }
//     }
//     return std::make_pair(imin, imax+1);
// }

std::pair<double, double> GenSycl::ImpactData::span(double nsigma) const
{
    int ncount = -1;
    double tmin=0, tmax=0;
    for (auto diff : m_diffusions) {
        ++ncount;
        
        auto td = diff->time_desc();

        const double ltmin = td.center - td.sigma*nsigma;
        const double ltmax = td.center + td.sigma*nsigma;
        if (!ncount) {
            tmin = ltmin;
            tmax = ltmax;
            continue;
        }
        tmin = std::min(tmin, ltmin);
        tmax = std::max(tmax, ltmax);
    }        
    return std::make_pair(tmin,tmax);
}
