/*
 *            Copyright 2009-2017 The VOTCA Development Team
 *                       (http://www.votca.org)
 *
 *      Licensed under the Apache License, Version 2.0 (the "License")
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *              http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Overload of uBLAS prod function with MKL/GSL implementations

#include <votca/xtp/gwbse.h>
#include <votca/tools/constants.h>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <votca/xtp/numerical_integrations.h>
#include <votca/ctp/logger.h>
#include <votca/xtp/qmpackagefactory.h>
#include <votca/tools/linalg.h>




using boost::format;
using namespace boost::filesystem;

namespace votca {
    namespace xtp {
        namespace ub = boost::numeric::ublas;
 

        // +++++++++++++++++++++++++++++ //
        // GWBSE MEMBER FUNCTIONS        //
        // +++++++++++++++++++++++++++++ //

        void GWBSE::CleanUp() {

        }

        void GWBSE::Initialize(Property* options) {

#if (GWBSE_DOUBLE)
            CTP_LOG(ctp::logDEBUG, *_pLog) << " Compiled with full double support" << flush;
#else
            CTP_LOG(ctp::logDEBUG, *_pLog) << " Compiled with float/double mixture (standard)" << flush;
#endif

            string key = Identify();

            // getting level ranges 
            _ranges=options->ifExistsReturnElseReturnDefault<string>(key + ".ranges","default");
           
            // now check validity, and get rpa, qp, and bse level ranges accordingly


            if (_ranges == "factor") {
                // get factors
                _rpamaxfactor = options->get(key + ".rpamax").as<double> ();
                _qpminfactor = options->get(key + ".qpmin").as<double> ();
                _qpmaxfactor = options->get(key + ".qpmax").as<double> ();
                _bseminfactor = options->get(key + ".bsemin").as<double> ();
                _bsemaxfactor = options->get(key + ".bsemax").as<double> ();
            } else if (_ranges == "explicit") {
                //get explicit numbers
                _rpamax = options->get(key + ".rpamax").as<unsigned int> ();
                _qpmin = options->get(key + ".qpmin").as<unsigned int> ();
                _qpmax = options->get(key + ".qpmax").as<unsigned int> ();
                _bse_vmin = options->get(key + ".bsemin").as<unsigned int> ();
                _bse_cmax = options->get(key + ".bsemax").as<unsigned int> ();
            } else if (_ranges == "" || _ranges=="default") {
                _ranges = "default";
            } else if(_ranges=="full"){
                _ranges="full";
            } else {
                cerr << "\nSpecified range option " << _ranges << " invalid. ";
                throw std::runtime_error("\nValid options are: default,factor,explicit,full");
            }

            _bse_nmax = options->ifExistsReturnElseReturnDefault<int>(key + ".exctotal", 25);
            _bse_nprint=options->ifExistsReturnElseReturnDefault<int>(key + ".print", 25);
            _fragA=options->ifExistsReturnElseReturnDefault<int>(key + ".fragment", -1);
            
            string BSEtype=options->ifExistsReturnElseReturnDefault<string>(key + ".BSEtype", "TDA");
            
            if (BSEtype == "full") {
                _do_full_BSE = true;
                CTP_LOG(ctp::logDEBUG, *_pLog) << " BSE type: full" << flush;
               }
           else {
                _do_full_BSE = false;
                CTP_LOG(ctp::logDEBUG, *_pLog) << " BSE type: TDA" << flush;
            }
            
            _openmp_threads = options->ifExistsReturnElseReturnDefault<int>(key + ".openmp",0);

            
            if (options->exists(key + ".vxc")) {
                _doVxc = options->ifExistsReturnElseThrowRuntimeError<bool>(key + ".vxc.dovxc");
                if (_doVxc) {
                    _functional = options->ifExistsReturnElseThrowRuntimeError<string>(key + ".vxc.functional");
                    _grid = options->ifExistsReturnElseReturnDefault<string>(key + ".vxc.grid", "medium");
                }
            }
            
            _gwbasis_name = options->ifExistsReturnElseThrowRuntimeError<string>(key + ".gwbasis");
            _dftbasis_name = options->ifExistsReturnElseThrowRuntimeError<string>(key + ".dftbasis");
            
            _shift = options->ifExistsReturnElseThrowRuntimeError<double>(key + ".shift");
            _qp_limit = options->ifExistsReturnElseReturnDefault<double>(key + ".qp_limit", 0.00001);//convergence criteria for qp iteration [Hartree]]
            _qp_max_iterations = options->ifExistsReturnElseReturnDefault<int>(key + ".qp_max_iterations", 20);//convergence criteria for qp iteration [Hartree]]

            _shift_limit = options->ifExistsReturnElseReturnDefault<double>(key + ".shift_limit", 0.00001);//convergence criteria for shift it
            _iterate_qp = false;
            string _shift_type =options->ifExistsReturnElseThrowRuntimeError<string>(key + ".shift_type");
            if (_shift_type != "fixed"){ _iterate_qp = true;}
            CTP_LOG(ctp::logDEBUG, *_pLog) << " Shift: " << _shift_type << flush;
            CTP_LOG(ctp::logDEBUG, *_pLog) << " qp_limit [Hartree]: " << _qp_limit << flush;
            if (_iterate_qp) {
                CTP_LOG(ctp::logDEBUG, *_pLog) << " shift_limit [Hartree]: " << _shift_limit << flush;
            }
            _min_print_weight = options->ifExistsReturnElseReturnDefault<double>(key + ".bse_print_weight", 0.2);//print exciton WF composition weight larger that thin minimum

              // setting some defaults
            _do_qp_diag = false;
            _do_bse_singlets = false;
            _do_bse_triplets = false;
            // possible tasks
            // diagQP, singlets, triplets, all, ibse
            string _tasks_string = options->ifExistsReturnElseThrowRuntimeError<string>(key + ".tasks");
            if (_tasks_string.find("all") != std::string::npos) {
                _do_qp_diag = true;
                _do_bse_singlets = true;
                _do_bse_triplets = true;
            }
            if (_tasks_string.find("qpdiag") != std::string::npos) _do_qp_diag = true;
            if (_tasks_string.find("singlets") != std::string::npos) _do_bse_singlets = true;
            if (_tasks_string.find("triplets") != std::string::npos) _do_bse_triplets = true;
            _store_eh_interaction = false;
            _do_bse_diag = true;
            // special construction for ibse mode
            if (_tasks_string.find("igwbse") != std::string::npos) {
                _do_qp_diag = false; // no qp diagonalization
                _do_bse_diag = false; // no diagonalization of BSE Hamiltonian
                _store_eh_interaction = true;

            }

            // possible storage 
            // qpPert, qpdiag_energies, qp_diag_coefficients, bse_singlet_energies, bse_triplet_energies, bse_singlet_coefficients, bse_triplet_coefficients
            _store_qp_pert = true;
            
            
            _store_qp_diag = false;
            _store_bse_triplets=false;
            _store_bse_singlets=false;
            string _store_string = options->ifExistsReturnElseThrowRuntimeError<string>(key + ".store");
            if ((_store_string.find("all") != std::string::npos) || (_store_string.find("") != std::string::npos)) {
                // store according to tasks choice
                if (_do_qp_diag) _store_qp_diag = true;
                if (_do_bse_singlets && _do_bse_diag) _store_bse_singlets = true;
                if (_do_bse_triplets && _do_bse_diag) _store_bse_triplets = true;
            }
            if (_store_string.find("qpdiag") != std::string::npos) _store_qp_diag = true;
            if (_store_string.find("singlets") != std::string::npos) _store_bse_singlets = true;
            if (_store_string.find("triplets") != std::string::npos) _store_bse_triplets = true;
            if (_store_string.find("ehint") != std::string::npos) _store_eh_interaction = true;


            CTP_LOG(ctp::logDEBUG, *_pLog) << " Tasks: " << flush;
            if (_do_qp_diag) {
                CTP_LOG(ctp::logDEBUG, *_pLog) << " qpdiag " << flush;
            }
            if (_do_bse_singlets) {
                CTP_LOG(ctp::logDEBUG, *_pLog) << " singlets " << flush;
            }
            if (_do_bse_triplets) {
                CTP_LOG(ctp::logDEBUG, *_pLog) << " triplets " << flush;
            }
            CTP_LOG(ctp::logDEBUG, *_pLog) << " Store: " << flush;
            if (_store_qp_diag) {
                CTP_LOG(ctp::logDEBUG, *_pLog) << " qpdiag " << flush;
            }
            if (_store_bse_singlets) {
                CTP_LOG(ctp::logDEBUG, *_pLog) << " singlets " << flush;
            }
            if (_store_bse_triplets) {
                CTP_LOG(ctp::logDEBUG, *_pLog) << " triplets " << flush;
            }
            if (_store_eh_interaction) {
                CTP_LOG(ctp::logDEBUG, *_pLog) << " ehint " << flush;
            }

            return;
        }

        void GWBSE::addoutput(Property *_summary) {
            
            const double hrt2ev = tools::conv::hrt2ev;
            Property *_gwbse_summary = &_summary->add("GWBSE", "");
            _gwbse_summary->setAttribute("units", "eV");
            _gwbse_summary->setAttribute("DeltaHLGap", (format("%1$+1.6f ") % (_shift * hrt2ev)).str());

            _gwbse_summary->setAttribute("DFTEnergy", (format("%1$+1.6f ") % _orbitals->getQMEnergy()).str());
            int printlimit = _bse_nprint; //I use this to determine how much is printed, I do not want another option to pipe through

            Property *_dft_summary = &_gwbse_summary->add("dft", "");
            _dft_summary->setAttribute("HOMO", _homo);
            _dft_summary->setAttribute("LUMO", _homo + 1);
            int begin = _homo - printlimit;
            int end = _homo + printlimit + 1;
            if (begin < 0) {
                begin = 0;
                end = 2 * _homo + 1;
            }
            for (int state = begin; state < end; state++) {

                Property *_level_summary = &_dft_summary->add("level", "");
                _level_summary->setAttribute("number", state);
                _level_summary->add("dft_energy", (format("%1$+1.6f ") % ((_orbitals->MOEnergies())(_qpmin + state) * hrt2ev)).str());
                _level_summary->add("gw_energy", (format("%1$+1.6f ") % (_qp_energies(_qpmin + state) * hrt2ev)).str());

                if (_do_qp_diag) {
                    //cout << "_do_qp_diag" <<_do_qp_diag<<endl;
                    _level_summary->add("qp_energy", (format("%1$+1.6f ") % (_qp_diag_energies(_qpmin + state) * hrt2ev)).str());
                }
            }

            if (_do_bse_singlets) {
                Property *_singlet_summary = &_gwbse_summary->add("singlets", "");
                for (int state = 0; state < printlimit; ++state) {
                    Property *_level_summary = &_singlet_summary->add("level", "");
                    _level_summary->setAttribute("number", state + 1);
                    _level_summary->add("omega", (format("%1$+1.6f ") % (_bse_singlet_energies(state) * hrt2ev)).str());
                    if (_orbitals->hasTransitionDipoles()) {

                        const tools::vec & dipoles = (_orbitals->TransitionDipoles())[state];
                        double f =2* dipoles*dipoles * _bse_singlet_energies(state) / 3.0;

                        _level_summary->add("f", (format("%1$+1.6f ") % f).str());
                        Property *_dipol_summary = &_level_summary->add("Trdipole", (format("%1$+1.4f %2$+1.4f %3$+1.4f") % dipoles.getX() % dipoles.getY() % dipoles.getZ()).str());
                        _dipol_summary->setAttribute("unit", "e*bohr");
                        _dipol_summary->setAttribute("gauge", "length");
                    }
                }
            }
            if (_do_bse_triplets) {
                Property *_triplet_summary = &_gwbse_summary->add("triplets", "");
                for (int state = 0; state < printlimit; ++state) {

                    Property *_level_summary = &_triplet_summary->add("level", "");
                    _level_summary->setAttribute("number", state + 1);
                    _level_summary->add("omega", (format("%1$+1.6f ") % (_bse_triplet_energies(state) * hrt2ev)).str());
                }
            }
            return;
        }

        /* 
         *    Many-body Green's fuctions theory implementation
         * 
         *  data required from orbitals file
         *  - atomic coordinates
         *  - DFT molecular orbitals (energies and coeffcients)
         *  - DFT exchange-correlation potential matrix in atomic orbitals
         *  - number of electrons, number of levels 
         * 
         
         */

        bool GWBSE::Evaluate() {

            // set the parallelization 
#ifdef _OPENMP
            if (_openmp_threads > 0){ 
            omp_set_num_threads(_openmp_threads);
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp()  << " Using "<< omp_get_max_threads()<<" threads" << flush;
            }
#endif
            /* check which QC program was used for the DFT run 
             * -> implicit info about MO coefficient storage order 
             */
            string _dft_package = _orbitals->getQMpackage();
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " DFT data was created by " << _dft_package << flush;

            std::vector<ctp::QMAtom*> _atoms;
            for(const auto& atom:_orbitals->QMAtoms()){
                            if(!atom->from_environment){
                            _atoms.push_back(atom);
                            }
                        }
            // load DFT basis set (element-wise information) from xml file
            BasisSet dftbs;

            if (_dftbasis_name != _orbitals->getDFTbasis()) {
                throw std::runtime_error("Name of the Basisset from .orb file: " + _orbitals->getDFTbasis() 
                + " and from GWBSE optionfile " + _dftbasis_name + " do not agree.");
            }

            dftbs.LoadBasisSet(_dftbasis_name);
            _orbitals->setDFTbasis(_dftbasis_name);
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Loaded DFT Basis Set " << _dftbasis_name << flush;

            // fill DFT AO basis by going through all atoms 
           
            _dftbasis.AOBasisFill(&dftbs, _atoms, _fragA);
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Filled DFT Basis of size " << _dftbasis.AOBasisSize() << flush;
            if (_dftbasis._AOBasisFragB > 0) {
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " FragmentA size " << _dftbasis._AOBasisFragA << flush;
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " FragmentB size " << _dftbasis._AOBasisFragB << flush;
            }

            /* Preparation of calculation parameters:
             *  - number of electrons -> index of HOMO
             *  - number of levels 
             *  - highest level considered in RPA
             *  - lowest and highest level considered in GWA
             *  - lowest and highest level considered in BSE 
             *  - number of excitations calculates in BSE
             */

            // convert _rpamax if needed 
            _homo = _orbitals->getNumberOfElectrons() - 1; // indexed from 0
            _rpamin = 0; // lowest index occ min(gwa%mmin, screening%nsum_low) ! always 1
            if (_ranges == "default" || _ranges=="full") {
                _rpamax = _orbitals->getNumberOfLevels() - 1; // total number of levels
            } else if (_ranges == "factor") {
                _rpamax = _rpamaxfactor * _orbitals->getNumberOfLevels() - 1; // total number of levels
            }
            if(_rpamax>_orbitals->getNumberOfLevels() - 1){
                _rpamax=_orbitals->getNumberOfLevels() - 1;
            }
            // convert _qpmin and _qpmax if needed
            if (_ranges == "default") {
                _qpmin = 0; // indexed from 0
                _qpmax = 2 * _homo + 1; // indexed from 0
            } else if (_ranges == "factor") {
                if(_orbitals->getNumberOfElectrons() - int( _qpminfactor * _orbitals->getNumberOfElectrons()) - 1<0){
                    _qpmin=0;
                }
                else{
                _qpmin = _orbitals->getNumberOfElectrons() - int( _qpminfactor * _orbitals->getNumberOfElectrons()) - 1;
                }
                _qpmax = _orbitals->getNumberOfElectrons() + int( _qpmaxfactor * _orbitals->getNumberOfElectrons()) - 1;
            } else if (_ranges == "explicit") {
                _qpmin -= 1;
                _qpmax -= 1;
            }else if(_ranges=="full"){
                _qpmin = 0;
                _qpmax= _orbitals->getNumberOfLevels() - 1;
            }
            if(_qpmax>unsigned(_orbitals->getNumberOfLevels() - 1)){
                _qpmax=_orbitals->getNumberOfLevels() - 1;
            }

            // set BSE band range indices 
            // anything else would be stupid!
            _bse_vmax = _homo;
            _bse_cmin = _homo + 1;

            if (_ranges == "default") {
                _bse_vmin = 0; // indexed from 0
                _bse_cmax = 2 * _homo + 1; // indexed from 0
            } else if (_ranges == "factor") {
                _bse_vmin = _orbitals->getNumberOfElectrons() - int( _bseminfactor * _orbitals->getNumberOfElectrons()) - 1;
                if(_orbitals->getNumberOfElectrons() - int( _bseminfactor * _orbitals->getNumberOfElectrons()) - 1<0){
                     _bse_vmin = 0;
                }
                _bse_cmax = _orbitals->getNumberOfElectrons() + int( _bsemaxfactor * _orbitals->getNumberOfElectrons()) - 1;
            } else if (_ranges == "explicit") {
                _bse_vmin -= 1;
                _bse_cmax -= 1;
            } else if(_ranges=="full"){
                _bse_vmin = 0;
                _bse_cmax = _orbitals->getNumberOfLevels() - 1;
            }
             if(_bse_cmax>unsigned(_orbitals->getNumberOfLevels() - 1)){
                _bse_cmax=_orbitals->getNumberOfLevels() - 1;
            }
            
            _bse_vtotal = _bse_vmax - _bse_vmin + 1;
            _bse_ctotal = _bse_cmax - _bse_cmin + 1;
            _bse_size = _bse_vtotal * _bse_ctotal;

            // indexing info BSE vector index to occupied/virtual orbital
            for (unsigned _v = 0; _v < _bse_vtotal; _v++) {
                for (unsigned _c = 0; _c < _bse_ctotal; _c++) {
                    _index2v.push_back(_bse_vmin + _v);
                    _index2c.push_back(_bse_cmin + _c);
                }
            }

            // some QP - BSE consistency checks are required
            if (_bse_vmin < _qpmin) _qpmin = _bse_vmin;
            if (_bse_cmax > _qpmax) _qpmax = _bse_cmax;
            
            
            _qptotal = _qpmax - _qpmin + 1;
            if (_bse_nmax > int(_bse_size) || _bse_nmax < 0) _bse_nmax = int(_bse_size);
            if (_bse_nprint > _bse_nmax) _bse_nprint = _bse_nmax;

            // store information in _orbitals for later use
            _orbitals->setRPAindices(_rpamin, _rpamax);
            _orbitals->setGWAindices(_qpmin, _qpmax);
            _orbitals->setBSEindices(_bse_vmin, _bse_vmax, _bse_cmin, _bse_cmax, _bse_nmax);

            // information for hybrid DFT

            _ScaHFX = -1;


            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Set RPA level range [" << _rpamin + 1 << ":" << _rpamax + 1 << "]" << flush;
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Set QP  level range [" << _qpmin + 1 << ":" << _qpmax + 1 << "]" << flush;
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Set BSE level range occ[" << _bse_vmin + 1 << ":" << _bse_vmax + 1 << "]  virt[" << _bse_cmin + 1 << ":" << _bse_cmax + 1 << "]" << flush;
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " BSE Hamiltonian has size " <<_bse_size<<"x"<<_bse_size << flush;


            // process the DFT data
            // a) form the expectation value of the XC functional in MOs
            // we do not link them because we have to reorder them afterwards and orbitals file has them in the order of the qmpackage
            //make copy of orbitals MOs because we reorder them
             _dft_orbitals = _orbitals->MOCoefficients(); //
            _ScaHFX = _orbitals->getScaHFX();
            {// this bracket is there so that _vx_ao falls out of scope, like it more than resize
                ub::matrix<double> _vxc_ao;
                if (_orbitals->hasAOVxc()) {
                    if (_doVxc) {
                        CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << "There is already a Vxc matrix loaded from DFT, did you maybe run a DFT code with outputVxc?\n I will take the external implementation" << flush;
                        _doVxc = false;
                    }
                    if (_dft_package == "gaussian") {
                        // we have to do some cartesian -> spherical transformation for Gaussian
                        const ub::matrix<double>& vxc_cart = _orbitals->AOVxc();
                        //cout<< vxc_cart.size1()<<"x"<<vxc_cart.size2()<<endl;
                        ub::matrix<double> _carttrafo;
                        _dftbasis.getTransformationCartToSpherical(_dft_package, _carttrafo);
                        //cout<< _carttrafo.size1()<<"x"<<_carttrafo.size2()<<endl;

                        ub::matrix<double> _temp = ub::prod(_carttrafo, vxc_cart);
                        _vxc_ao = ub::prod(_temp, ub::trans(_carttrafo));

                    } else {
                        _vxc_ao = _orbitals->AOVxc();
                    }
                } else if (_doVxc) {

                    NumericalIntegration _numint;
                    _numint.setXCfunctional(_functional);
                    double ScaHFX_temp = _numint.getExactExchange(_functional);
                    if (ScaHFX_temp != _ScaHFX) {
                        throw std::runtime_error((boost::format("GWBSE exact exchange a=%s differs from qmpackage exact exchange a=%s, probably your functionals are inconsistent") % ScaHFX_temp % _ScaHFX).str());
                    }
                    _numint.GridSetup(_grid, &dftbs, _atoms,&_dftbasis);
                    CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Setup grid for integration with gridsize: " << _grid <<" with "<<_numint.getGridSize()<< " points, divided into "<<_numint.getBoxesSize()<<" boxes"<<flush;
                    _dftbasis.ReorderMOs(_dft_orbitals, _dft_package, "xtp");

                    CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Converted DFT orbital coefficient order from " << _dft_package << " to XTP" << flush;
                    CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Integrating Vxc in VOTCA with functional " << _functional << flush;
                    ub::matrix<double> DMAT = _orbitals->DensityMatrixGroundState(_dft_orbitals);
                    
                    _vxc_ao = _numint.IntegrateVXC(DMAT);
                    CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Calculated Vxc in VOTCA" << flush;

                } else {
                    throw std::runtime_error("So your DFT data contains no Vxc, if you want to proceed use the dovxc option.");
                }

                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Set hybrid exchange factor: " << _ScaHFX << flush;

                // now get expectation values but only for those in _qpmin:_qpmax range
                ub::matrix<double> _mos = ub::project(_dft_orbitals, ub::range(_qpmin, _qpmax + 1), ub::range(0, _dftbasis.AOBasisSize()));
                
                ub::matrix<double> _temp = ub::prod(_vxc_ao, ub::trans(_mos));
                _vxc = ub::prod(_mos, _temp);
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Calculated exchange-correlation expectation values " << flush;

                // b) reorder MO coefficients depending on the QM package used to obtain the DFT data
                if (_dft_package != "xtp" && !_doVxc) {
                    _dftbasis.ReorderMOs(_dft_orbitals, _dft_package, "xtp");
                    CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Converted DFT orbital coefficient order from " << _dft_package << " to XTP" << flush;
                }
            }

            /// ------- actual calculation begins here -------


            // load auxiliary GW basis set (element-wise information) from xml file
            BasisSet gwbs;
            gwbs.LoadBasisSet(_gwbasis_name);
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Loaded GW Basis Set " << _gwbasis_name << flush;

            // fill auxiliary GW AO basis by going through all atoms
            AOBasis gwbasis;
            gwbasis.AOBasisFill(&gwbs, _atoms);
            _orbitals->setGWbasis(_gwbasis_name);
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Filled GW Basis of size " << gwbasis.AOBasisSize() << flush;

            /* 
             * for the representation of 2-point functions with the help of the 
             * auxiliary GW basis, its AO overlap matrix is required.
             * cf. M. Rohlfing, PhD thesis, ch. 3 
             */
            AOOverlap _gwoverlap;
            // initialize overlap matrix
            _gwoverlap.Initialize(gwbasis.AOBasisSize());
            // Fill overlap
            _gwoverlap.Fill(gwbasis);


            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Filled GW Overlap matrix of dimension: " << _gwoverlap.Matrix().size1() << flush;
          
            // check eigenvalues of overlap matrix, if too small basis might have linear dependencies
            ub::vector<double> _eigenvalues;
            ub::matrix<double> _eigenvectors;
            linalg_eigenvalues(_gwoverlap.Matrix(), _eigenvalues, _eigenvectors);
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Smallest eigenvalue of GW Overlap matrix : " << _eigenvalues[0] << flush;


            /*
             *  for the calculation of Coulomb and exchange term in the self
             *  energy and electron-hole interaction, the Coulomb interaction 
             *  is represented using the auxiliary GW basis set.
             *  Here, we need to prepare the Coulomb matrix expressed in 
             *  the AOs of the GW basis 
             */

            // get Coulomb matrix as AOCoulomb
            AOCoulomb _gwcoulomb;
            // initialize Coulomb matrix
            _gwcoulomb.Initialize(gwbasis.AOBasisSize());
            // Fill Coulomb matrix
            _gwcoulomb.Fill(gwbasis);
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Filled GW Coulomb matrix of dimension: " << _gwcoulomb.Matrix().size1() << flush;
          

            // PPM is symmetric, so we need to get the sqrt of the Coulomb matrix
            ub::matrix<double> _gwoverlap_inverse; // will also be needed in PPM itself
            ub::matrix<double> _gwoverlap_cholesky_inverse; // will also be needed in PPM itself
           
            _gwcoulomb.Symmetrize(_gwoverlap, gwbasis, _gwoverlap_inverse, _gwoverlap_cholesky_inverse);
            ub::matrix<double> _gwoverlap_cholesky_inverse_trans=ub::trans(_gwoverlap_cholesky_inverse);// for performance reasons
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Prepared GW Coulomb matrix for symmetric PPM " << flush;

            /* calculate 3-center integrals,  convoluted with DFT eigenvectors
             * 
             *  M_mn(beta) = \int{ \psi^DFT_m(r) \phi^GW_beta(r) \psi^DFT_n d3r  }
             *             = \sum_{alpha,gamma} { c_m,alpha c_n,gamma \int {\phi^DFT_alpha(r) \phi^GW_beta(r) \phi^DFT_gamma(r) d3r}  }
             *
             *  cf. M. Rohlfing, PhD thesis, ch. 3.2
             * 
             */

            // --- prepare a vector (gwdacay) of matrices (orbitals, orbitals) as container => M_mn
            // prepare 3-center integral object

            TCMatrix _Mmn;
            _Mmn.Initialize(gwbasis.AOBasisSize(), _rpamin, _qpmax, _rpamin, _rpamax);
            _Mmn.Fill(gwbasis, _dftbasis, _dft_orbitals);
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Calculated Mmn_beta (3-center-overlap x orbitals)  " << flush;

            // for use in RPA, make a copy of _Mmn with dimensions (1:HOMO)(gwabasissize,LUMO:nmax)
            TCMatrix _Mmn_RPA;
            _Mmn_RPA.Initialize(gwbasis.AOBasisSize(), _rpamin, _homo, _homo + 1, _rpamax);
            RPA_prepare_threecenters(_Mmn_RPA, _Mmn, gwbasis, _gwoverlap, _gwoverlap_inverse);
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Prepared Mmn_beta for RPA  " << flush;


            // make _Mmn_RPA symmetric for use in RPA
            _Mmn_RPA.Symmetrize(_gwcoulomb.Matrix());
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Symmetrize Mmn_beta for RPA  " << flush;

            // make _Mmn symmetric for use in self-energy calculation
            _Mmn.Symmetrize(_gwcoulomb.Matrix());
            CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Symmetrize Mmn_beta for self-energy  " << flush;

            // fix the frequencies for PPM
            _screening_freq = ub::zero_matrix<double>(2, 2); // two frequencies
            //first one
            _screening_freq(0, 0) = 0.0; // real part
            _screening_freq(0, 1) = 0.0; // imaginary part
            //second one
            _screening_freq(1, 0) = 0.0; // real part
            _screening_freq(1, 1) = 0.5; // imaginary part  //hartree

            // one entry to epsilon for each frequency
            _epsilon.resize(_screening_freq.size1());

            /* for automatic iteration of _shift, we need to
             * - make a copy of _Mmn
             * - calculate eps
             * - construct ppm
             * - threecenters for sigma
             * - sigma_x
             * - sigma_c 
             * - test for convergence
             * 
             */
            
            //initialize _qp_energies;
            //shift unoccupied levels by the shift
            _qp_energies=ub::zero_vector<double>(_orbitals->getNumberOfLevels());
            for (size_t i=0;i<_qp_energies.size();++i){
                    _qp_energies(i)=_orbitals->MOEnergies()(i);
                if(i>_homo){
                    _qp_energies(i)+=_shift;
                }                
            }
            
           
            TCMatrix _Mmn_backup;
            if (_iterate_qp) {

                // make copy of _Mmn, memory++

                _Mmn_backup.Initialize(gwbasis.AOBasisSize(), _rpamin, _qpmax, _rpamin, _rpamax);
                int _mnsize = _Mmn_backup.get_mtot();
                for (int _i = 0; _i < _mnsize; _i++) {
                    _Mmn_backup[ _i ] = _Mmn[ _i ];
                }
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Made backup of _Mmn  " << flush;
            }
            _qp_converged=false;
            _qp_iteration=0;
            while (!_qp_converged) {
               _qp_iteration++;
                // for symmetric PPM, we can initialize _epsilon with the overlap matrix!
                for (unsigned _i_freq = 0; _i_freq < _screening_freq.size1(); _i_freq++) {
                    _epsilon[ _i_freq ] = _gwoverlap.Matrix();
                }
               
                
                // determine epsilon from RPA
                RPA_calculate_epsilon(_Mmn_RPA);
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Calculated epsilon via RPA  " << flush;

                 

                // construct PPM parameters
                PPM_construct_parameters(_gwoverlap_cholesky_inverse,_gwoverlap_cholesky_inverse_trans);
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Constructed PPM parameters  " << flush;
                if (!_iterate_qp){
                    _qp_converged = true;
                    _gwoverlap.Matrix().resize(0, 0);
                    _gwoverlap_cholesky_inverse.resize(0,0);
                    _gwoverlap_cholesky_inverse_trans.resize(0,0);
                    _Mmn_RPA.Cleanup();
                    CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Cleaned up Overlap and MmnRPA" << flush;
                }
                // prepare threecenters for Sigma
                sigma_prepare_threecenters(_Mmn);
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Prepared threecenters for sigma  " << flush;

                // calculate exchange part of sigma
                sigma_x_setup(_Mmn);
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Calculated exchange part of Sigma  " << flush;

           
                // calculate correlation part of sigma
                sigma_c_setup(_Mmn);
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Calculated correlation part of Sigma  " << flush;

                // restore _Mmn, if shift has not converged
                if (_iterate_qp && !_qp_converged) {
                    int _mnsize = _Mmn_backup.get_mtot();
                    for (int _i = 0; _i < _mnsize; _i++) {
                        _Mmn[ _i ] = _Mmn_backup[ _i ];
                    }

                }

            }

            // free unused variable if shift is iterated
            if (_iterate_qp) {
                _gwoverlap.Matrix().resize(0, 0);
                 _gwoverlap_cholesky_inverse.resize(0,0);
                _gwoverlap_cholesky_inverse_trans.resize(0,0);
                _Mmn_RPA.Cleanup();
                _Mmn_backup.Cleanup();
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Cleaned up Overlap, MmnRPA and Mmn_backup " << flush;
            }
            // free no longer required three-center matrices in _Mmn
            // max required is _bse_cmax (could be smaller than _qpmax)
            _Mmn.Prune(gwbasis.AOBasisSize(), _bse_vmin, _bse_cmax);


            const ub::vector<double>& _dft_energies=_orbitals->MOEnergies();
            // Output of quasiparticle energies after all is done:
           
            CTP_LOG(ctp::logINFO, *_pLog) << (format("  ====== Perturbative quasiparticle energies (Hartree) ====== ")).str() << flush;
            CTP_LOG(ctp::logINFO, *_pLog) << (format("   DeltaHLGap = %1$+1.6f Hartree") % _shift).str() << flush;
            
            for (unsigned _i = 0; _i < _qptotal; _i++) {
                if ((_i + _qpmin) == _homo) {
                    CTP_LOG(ctp::logINFO, *_pLog) << (format("  HOMO  = %1$4d DFT = %2$+1.4f VXC = %3$+1.4f S-X = %4$+1.4f S-C = %5$+1.4f GWA = %6$+1.4f") 
                            % (_i + _qpmin + 1) % _dft_energies(_i + _qpmin) % _vxc(_i, _i) % _sigma_x(_i, _i) % _sigma_c(_i, _i) % _qp_energies(_i + _qpmin)).str() << flush;
                } else if ((_i + _qpmin) == _homo + 1) {
                    CTP_LOG(ctp::logINFO, *_pLog) << (format("  LUMO  = %1$4d DFT = %2$+1.4f VXC = %3$+1.4f S-X = %4$+1.4f S-C = %5$+1.4f GWA = %6$+1.4f")
                            % (_i + _qpmin + 1) % _dft_energies(_i + _qpmin) % _vxc(_i, _i) % _sigma_x(_i, _i) % _sigma_c(_i, _i) % _qp_energies(_i + _qpmin)).str() << flush;

                } else {
                    CTP_LOG(ctp::logINFO, *_pLog) << (format("  Level = %1$4d DFT = %2$+1.4f VXC = %3$+1.4f S-X = %4$+1.4f S-C = %5$+1.4f GWA = %6$+1.4f") 
                            % (_i + _qpmin + 1) % _dft_energies(_i + _qpmin) % _vxc(_i, _i) % _sigma_x(_i, _i) % _sigma_c(_i, _i) % _qp_energies(_i + _qpmin)).str() << flush;
                }
            }




            // store perturbative QP energy data in orbitals object (DFT, S_x,S_c, V_xc, E_qp) 
            if (_store_qp_pert) {
                ub::matrix<double>& _qp_energies_store = _orbitals->QPpertEnergies();
                _qp_energies_store.resize(_qptotal, 5);
                for (unsigned _i = 0; _i < _qptotal; _i++) {
                    _qp_energies_store(_i, 0) = _dft_energies(_i + _qpmin);
                    _qp_energies_store(_i, 1) = _sigma_x(_i, _i);
                    _qp_energies_store(_i, 2) = _sigma_c(_i, _i);
                    _qp_energies_store(_i, 3) = _vxc(_i, _i);
                    _qp_energies_store(_i, 4) = _qp_energies(_i + _qpmin);
                }
            }


            // constructing full quasiparticle Hamiltonian and diagonalize, if requested
            if (_do_qp_diag || _do_bse_singlets || _do_bse_triplets) {
                FullQPHamiltonian();
                if (_do_qp_diag) {
                    CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Full quasiparticle Hamiltonian  " << flush;
                    CTP_LOG(ctp::logINFO, *_pLog) << (format("  ====== Diagonalized quasiparticle energies (Hartree) ====== ")).str() << flush;
                    for (unsigned _i = 0; _i < _qptotal; _i++) {
                        if ((_i + _qpmin) == _homo) {
                            CTP_LOG(ctp::logINFO, *_pLog) << (format("  HOMO  = %1$4d PQP = %2$+1.4f DQP = %3$+1.4f ") % (_i + _qpmin + 1) % _qp_energies(_i + _qpmin) % _qp_diag_energies(_i)).str() << flush;
                        } else if ((_i + _qpmin) == _homo + 1) {
                            CTP_LOG(ctp::logINFO, *_pLog) << (format("  LUMO  = %1$4d PQP = %2$+1.4f DQP = %3$+1.4f ") % (_i + _qpmin + 1) % _qp_energies(_i + _qpmin) % _qp_diag_energies(_i)).str() << flush;

                        } else {
                            CTP_LOG(ctp::logINFO, *_pLog) << (format("  Level = %1$4d PQP = %2$+1.4f DQP = %3$+1.4f ") % (_i + _qpmin + 1) % _qp_energies(_i + _qpmin) % _qp_diag_energies(_i)).str() << flush;
                        }
                    }

                    // free memory

                    if (!_store_qp_diag) {
                        _qp_diag_coefficients.resize(0, 0);
                        _qp_diag_energies.resize(0);
                    }
                } // _do_qp_diag
            } // constructing full quasiparticle Hamiltonian


            // proceed only if BSE requested
            if (_do_bse_singlets || _do_bse_triplets) {

                // calculate direct part of eh interaction, needed for singlets and triplets
                BSE_d_setup(_Mmn);
                // add qp part to Eh_d
                BSE_Add_qp2H(_eh_d);
                // Only if verbose do we use this really
                if (votca::tools::globals::verbose) {
                    BSE_qp_setup();
                }

                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Direct part of e-h interaction " << flush;

                if (_do_full_BSE) {
                    BSE_d2_setup(_Mmn);
                    CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Direct part of e-h interaction RARC " << flush;
                }


                if (_do_bse_triplets && _do_bse_diag) {
                    BSE_solve_triplets();
                    CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Solved BSE for triplets " << flush;
                    // analyze and report results
                    BSE_analyze_triplets();

                } // do_triplets



                // constructing electron-hole interaction for BSE
                if (_do_bse_singlets) {
                    // calculate exchange part of eh interaction, only needed for singlets
                    BSE_x_setup(_Mmn);
                    CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Exchange part of e-h interaction " << flush;
                }

                if (_do_bse_singlets && _do_bse_diag) {

                    if (_do_full_BSE) {
                        BSE_solve_singlets_BTDA();
                        CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Solved full BSE for singlets " << flush;
                        BSE_analyze_singlets_BTDA();
                    } else {
                        BSE_solve_singlets();
                        CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " Solved BSE for singlets " << flush;
                        BSE_analyze_singlets();
                    }

                    if (!_store_eh_interaction) {
                        _eh_d.resize(0, 0);
                        _eh_x.resize(0, 0);
                    }
                    
                }
            }
                CTP_LOG(ctp::logDEBUG, *_pLog) << ctp::TimeStamp() << " GWBSE calculation finished " << flush;
                return true;
        }
    }
};
