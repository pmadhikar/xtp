#include <votca/xtp/polarseg.h>
#include <boost/format.hpp>
#include <votca/xtp/dmaspace.h>
#include <votca/xtp/molpolengine.h>


namespace votca { namespace xtp {

    
    
PolarSeg::PolarSeg(int id, vector<APolarSite*> &psites) 
    : _id(id), _is_charged(true), _is_polarizable(true), 
      _indu_cg_site(NULL), _perm_cg_site(NULL) {
    PolarFrag *pfrag = this->AddFragment("NN");
    for (unsigned int i = 0; i < psites.size(); ++i) {
        push_back(psites[i]);
        pfrag->push_back(psites[i]);
    }
    this->CalcPos();
}



PolarSeg::PolarSeg(int id, vector< QMAtom* > &qmatoms)
    : _id(id), _is_charged(true), _is_polarizable(true), 
      _indu_cg_site(NULL), _perm_cg_site(NULL) {
    PolarFrag *pfrag = this->AddFragment("NN");
    for (unsigned int i = 0; i < qmatoms.size(); ++i) {
        APolarSite *site=new APolarSite(qmatoms[i]);
        push_back(site);
        pfrag->push_back(site);
    }
    this->CalcPos();
}


PolarSeg::PolarSeg(PolarSeg *templ, bool do_depolarize) {
    // NOTE Polar neighbours _nbs are not copied !
    for (unsigned int i = 0; i < templ->_pfrags.size(); ++i) {
        PolarFrag *ref_frag = templ->_pfrags[i];
        PolarFrag *new_frag = this->AddFragment(ref_frag->getName());
        for (PolarFrag::iterator pit = ref_frag->begin();
            pit != ref_frag->end(); ++pit) {
            APolarSite *new_site = new APolarSite(*pit, do_depolarize);
            new_frag->push_back(new_site);
            this->push_back(new_site);
        }
    }
    this->_id = templ->_id;
    this->_pos = templ->_pos;
    this->_is_charged = templ->_is_charged;
    this->_is_polarizable = templ->_is_polarizable;
    // Coarsegrained site (induced moments)
    if (templ->_indu_cg_site == NULL) {
        this->_indu_cg_site = NULL;
    }
    else {
        this->_indu_cg_site = new APolarSite(templ->_indu_cg_site, do_depolarize);
    }
    // Coarsegrained site (permanent moments)
    if (templ->_perm_cg_site == NULL) {
        this->_perm_cg_site = NULL;
    }
    else {
        this->_perm_cg_site = new APolarSite(templ->_perm_cg_site, do_depolarize);
    }
}
    
    
PolarSeg::~PolarSeg() {
   vector<APolarSite*> ::iterator pit;
   for (pit = begin(); pit < end(); ++pit) {         
       delete *pit;
   }
   clear();
   
   vector<PolarFrag*>::iterator fit;
   for (fit = _pfrags.begin(); fit < _pfrags.end(); ++fit) 
       delete *fit;
   _pfrags.clear();
   
   vector<PolarNb*>::iterator nit;
   for (nit = _nbs.begin(); nit < _nbs.end(); ++nit) 
       delete *nit;
   _nbs.clear();
   
   if (_perm_cg_site != NULL) delete _perm_cg_site;
   if (_indu_cg_site != NULL) delete _indu_cg_site;
}


PolarFrag *PolarSeg::AddFragment(string name) { 
    _pfrags.push_back(new PolarFrag(this, (int)_pfrags.size()+1, name)); 
    return _pfrags.back();
}


PolarNb *PolarSeg::AddNewPolarNb(PolarSeg *pseg) {
    PolarNb *new_nb = new PolarNb(pseg);
    this->_nbs.push_back(new_nb);
    return new_nb;
}


void PolarSeg::CalcPos() {    
    _pos = vec(0,0,0);    
    for (unsigned int i = 0; i < this->size(); ++i) {        
        _pos += (*this)[i]->getPos();        
    }
    if (this->size() > 0)
        _pos /= double(this->size());
}


double PolarSeg::CalcTotQ() {
    double Q = 0.0;
    for (unsigned int i = 0; i < this->size(); ++i) {
        Q += (*this)[i]->getQ00();
    }
    return Q;
}


vec PolarSeg::CalcTotD() {
    vec D = vec(0,0,0);
    for (unsigned int i = 0; i < this->size(); ++i) {
        if ((*this)[i]->getRank() > 0)
            D += (*this)[i]->getQ1();
        D += (*this)[i]->getQ00()* ((*this)[i]->getPos()-this->getPos());
    }
    return D;
}


void PolarSeg::Translate(const vec &shift) {    
    for (unsigned int i = 0; i < size(); ++i) {
        (*this)[i]->Translate(shift);
    }
    _pos += shift;
}


void PolarSeg::CalcIsCharged() {
    _is_charged = false;
    for (unsigned int i = 0; i < size(); ++i) {
        if ((*this)[i]->IsCharged()) _is_charged = true;
    }
    return;
}


void PolarSeg::CalcIsPolarizable() {
    _is_polarizable = false;
    for (unsigned int i = 0; i < size(); ++i) {
        if ((*this)[i]->IsPolarizable()) _is_polarizable = true;
    }
    return;
}


void PolarSeg::ClearPolarNbs() {
    vector<PolarNb*>::iterator nit;
    for (nit = _nbs.begin(); nit != _nbs.end(); ++nit) 
        delete *nit;
    _nbs.clear();
    return;
}


void PolarSeg::PrintPolarNbPDB(string outfile) {    
    FILE *out;
    out = fopen(outfile.c_str(),"w");
    PolarSeg::iterator pit;
    vector<PolarNb*>::iterator nit;
    for (pit = begin(); pit < end(); ++pit) {
        (*pit)->WritePdbLine(out, "CEN");
    }
    for (nit = _nbs.begin(); nit < _nbs.end(); ++nit) {
        PolarSeg *nb = (*nit)->getNb();
        nb->Translate((*nit)->getS());
        for (pit = nb->begin(); pit < nb->end(); ++pit) {
            (*pit)->WritePdbLine(out, "PNB");
        }
        nb->Translate(-1*(*nit)->getS());
    }
    fclose(out);
    return;
}


void PolarSeg::WriteMPS(string mpsfile, string tag) {    
    ofstream ofs;    
    ofs.open(mpsfile.c_str(), ofstream::out);
    if (!ofs.is_open()) {
        throw runtime_error("Bad file handle: " + mpsfile);
    }
    
    this->CalcPos();
    vec D = this->CalcTotD();
    ofs << (boost::format("! GENERATED BY VOTCA::XTP::%1$s\n") % tag);
    ofs << (boost::format("! N=%2$d Q[e]=%1$+1.7f D[e*nm]=%3$+1.7e %4$+1.7e %5$+1.7e\n")
        % CalcTotQ() % size() % D.getX() % D.getY() % D.getZ());
    ofs << boost::format("Units angstrom\n");

    iterator pit;
    for (pit = begin(); pit < end(); ++pit) {
        (*pit)->WriteMpsLine(ofs, "angstrom");
    }    
    ofs.close();    
}


void PolarSeg::GeneratePermInduCgSite(bool do_cg_polarizabilities) {
    // ATTENTION The same method appears in <PolarFrag>
    assert(!do_cg_polarizabilities && "NOT IMPLEMENTED, NOT NEEDED?");
    // Collapse multipole moments : position, rank L
    vec target_pos = _pos;
    int state = 0;
    int L = 2;
    vector<double> QCG(L*L+2*L+1, 0.0);  // permanent
    vector<double> uQCG(L*L+2*L+1, 0.0); // induced

    for (PolarSeg::iterator pit = begin();
        pit < end(); ++pit) {
        // PERMANENT MOMENTS
        // Convert real to complex moments            
        vector<double> Qlm = (*pit)->getQs(0);
        DMA::ComplexSphericalMoments Xlm(Qlm);
        // Shift moments
        DMA::MomentShift mshift;
        vec shift = target_pos - (*pit)->getPos();
        DMA::RegularSphericalHarmonics Clm(-shift);
        vector<DMA::cmplx> Xlm_shifted = mshift.Shift(Xlm, Clm);            
        // Convert complex to real moments & add to base
        DMA::RealSphericalMoments Qlm_shifted(Xlm_shifted);
        Qlm_shifted.AddToVector(QCG);

        // INDUCED MOMENTS
        // Convert real to complex moments
        vec u1 = (*pit)->getU1();
        vector<double> uQlm(L*L+2*L+1, 0.0);
        uQlm[1] = u1.getZ(); // NOTE order is z-x-y == 10-11c-11s
        uQlm[2] = u1.getX();
        uQlm[3] = u1.getY();
        DMA::ComplexSphericalMoments uXlm(uQlm);
        // Shift moments
        DMA::RegularSphericalHarmonics uClm(-shift);
        vector<DMA::cmplx> uXlm_shifted = mshift.Shift(uXlm, uClm);
        // Convert complex to real moments & add to base
        DMA::RealSphericalMoments uQlm_shifted(uXlm_shifted);
        uQlm_shifted.AddToVector(uQCG);
    }
        
    // Collapse polarizabilities
    votca::tools::matrix PCG;
    PCG.ZeroMatrix();
    
    // Zero induced dipole
    vec u1_cg_red = vec(0,0,0);
        
    // Generate new coarse-grained site from the above
    APolarSite *indu_cg_site = new APolarSite(this->getId(), "SGU");
    APolarSite *perm_cg_site = new APolarSite(this->getId(), "SGP");
    
    indu_cg_site->setResolution(APolarSite::coarsegrained);
    indu_cg_site->setPos(target_pos);
    indu_cg_site->setRank(L);
    
    perm_cg_site->setResolution(APolarSite::coarsegrained);
    perm_cg_site->setPos(target_pos);
    perm_cg_site->setRank(L);
    
    // ATTENTION Save INDUCED   moments as PERMANENT moments (<indu_cg_site>)
    // ATTENTION Save PERMANENT moments as PERMANENT moments (<perm_cg_site>)
    indu_cg_site->setQs(uQCG, state);
    indu_cg_site->setPs(PCG, state);
    indu_cg_site->setU1(u1_cg_red);
    indu_cg_site->Charge(state);
    
    perm_cg_site->setQs(QCG, state);
    perm_cg_site->setPs(PCG, state);
    perm_cg_site->setU1(u1_cg_red);
    perm_cg_site->Charge(state);
    
    // Deallocate previously allocated sites
    if (_indu_cg_site != NULL) delete _indu_cg_site;
    if (_perm_cg_site != NULL) delete _perm_cg_site;
    _indu_cg_site = indu_cg_site;
    _perm_cg_site = perm_cg_site;
    return;
}


void PolarSeg::Coarsegrain(bool cg_anisotropic) {
    // Reduce each polar fragment to a single polar site
    vector<APolarSite*> cg_sites;
    for (vector<PolarFrag*>::iterator fit = _pfrags.begin();
        fit < _pfrags.end(); ++fit) {        
        // Collapse multipole moments : position, rank L
        vec target_pos = (*fit)->CalcPosPolarWeights();
        int state = 0;
        int L = 2;
        vector<double> QCG(L*L+2*L+1, 0.0);  // permanent
        vector<double> uQCG(L*L+2*L+1, 0.0); // induced
        vec u1_cg_sum = vec(0,0,0);
        for (PolarFrag::iterator pit = (*fit)->begin();
            pit < (*fit)->end(); ++pit) {
            // PERMANENT MOMENTS
            // Convert real to complex moments            
            vector<double> Qlm = (*pit)->getQs(0);
            DMA::ComplexSphericalMoments Xlm(Qlm);
            // Shift moments
            DMA::MomentShift mshift;
            vec shift = target_pos - (*pit)->getPos();
            DMA::RegularSphericalHarmonics Clm(-shift);
            vector<DMA::cmplx> Xlm_shifted = mshift.Shift(Xlm, Clm);            
            // Convert complex to real moments & add to base
            DMA::RealSphericalMoments Qlm_shifted(Xlm_shifted);
            Qlm_shifted.AddToVector(QCG);
            
//            // Shift back (error check)
//            vector<double> Qlm_shifted_vector = Qlm_shifted.ToVector();
//            DMA::ComplexSphericalMoments Xlm_back(Qlm_shifted_vector);
//            DMA::RegularSphericalHarmonics Clm_back(shift);
//            vector<DMA::cmplx> Xlm_shifted_back = mshift.Shift(Xlm_back, Clm_back);
//            DMA::RealSphericalMoments Qlm_back(Xlm_shifted_back);
//            cout << endl << "restored";
//            Qlm_back.PrintReal();
            
            // INDUCED MOMENTS
            u1_cg_sum += (*pit)->getU1();
            // Convert real to complex moments
            vec u1 = (*pit)->getU1();
            vector<double> uQlm(L*L+2*L+1, 0.0);
            uQlm[1] = u1.getZ(); // NOTE order is z-x-y == 10-11c-11s
            uQlm[2] = u1.getX();
            uQlm[3] = u1.getY();
            DMA::ComplexSphericalMoments uXlm(uQlm);
            // Shift moments
            DMA::RegularSphericalHarmonics uClm(-shift);
            vector<DMA::cmplx> uXlm_shifted = mshift.Shift(uXlm, uClm);
            // Convert complex to real moments & add to base
            DMA::RealSphericalMoments uQlm_shifted(uXlm_shifted);
            uQlm_shifted.AddToVector(uQCG);
        }
        
        // Induced moments: discard moments of rank > 1
        vec u1_cg_red = vec(uQCG[2], uQCG[3], uQCG[1]); // order z-x-y to x-y-z
        assert(votca::tools::abs(u1_cg_sum-u1_cg_red) < 1e-9
            && "<PolarSeg::CoarseGrain> INDUCED MOMENTS TRAFO - ERROR");
        // Collapse polarizabilities
        MolPolEngine engine = MolPolEngine();
        votca::tools::matrix PCG = engine.CalculateMolPol(*(*fit), tools::globals::verbose && false);
        if (cg_anisotropic == false) {
            // Reduce to isotropic tensor
            votca::tools::matrix::eigensystem_t pcg_eigen;
            PCG.SolveEigensystem(pcg_eigen);
            double p11 = pcg_eigen.eigenvalues[0];
            double p22 = pcg_eigen.eigenvalues[1];
            double p33 = pcg_eigen.eigenvalues[2];
            double piso = pow(p11*p22*p33, 1./3.);
            PCG = votca::tools::matrix(
                vec(piso,0,0), vec(0,piso,0), vec(0,0,piso));
        }
        
        // Generate new coarse-grained site from the above
        APolarSite *cg_site = new APolarSite((*fit)->getId(), (*fit)->getName());
        cg_site->setResolution(APolarSite::coarsegrained);
        cg_site->setPos(target_pos);
        cg_site->setRank(L);
        cg_site->setQs(QCG, state);
        cg_site->setPs(PCG, state);
        cg_site->setU1(u1_cg_red);
        cg_site->Charge(state);
        cg_sites.push_back(cg_site);
        // Clear fragment & reload
        (*fit)->clear();
        (*fit)->push_back(cg_site);
    }
    
    // Clean up & reload
    assert(cg_sites.size() == _pfrags.size());
    vector<APolarSite*> ::iterator pit;
    for (pit = begin(); pit < end(); ++pit) delete *pit;
    clear();
    for (pit = cg_sites.begin(); pit < cg_sites.end(); ++pit) push_back(*pit);
    return;
}


}}
