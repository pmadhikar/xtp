#ifndef VOTCA_XTP_QMAPECALC_H
#define	VOTCA_XTP_QMAPECALC_H

#include <votca/xtp/pewald3d.h>
#include <votca/xtp/parallelxjobcalc.h>
#include <votca/xtp/xmapper.h>
#include <votca/xtp/xjob.h>
#include <votca/xtp/xinductor.h>
#include <votca/xtp/xinteractor.h>
#include <votca/xtp/gwbse.h>
#include <votca/xtp/qmapemachine.h>
#include <boost/format.hpp>

using boost::format;

namespace votca { namespace xtp {

   
class QMAPE : public ParallelXJobCalc< vector<Job*>, Job*, Job::JobResult >
{

public:

    QMAPE() {};
   ~QMAPE() {};
   
    string          Identify() { return "qmape"; }
    void            Initialize(Property *);

    void            CustomizeLogger(QMThread *thread);
    void            PreProcess(Topology *top);
    Job::JobResult  EvalJob(Topology *top, Job *job, QMThread *thread);
    XJob            ProcessInputString(Job *job, Topology *top, QMThread *thread);

private:
    
    // ======================================== //
    // MULTIPOLE ALLOCATION, XJOBS, ADD. OUTPUT //
    // ======================================== //

	string                         _xml_file;
	string                         _mps_table;
	string                         _polar_bg_arch;
	XMpsMap                        _mps_mapper;
	bool                           _pdb_check;
	bool                           _ptop_check;
    
    // ======================================== //
    // INDUCTION + ENERGY EVALUATION            //
    // ======================================== //

    // Induction, subthreading (-> base class)
    //bool                            _induce;
    //bool                            _induce_intra_pair;

    // Multipole Interaction parameters
    string                          _method;
    //bool                            _useCutoff;
    //double                          _cutoff1;
    //double                          _cutoff2;
    
    // QM Package options
    string                          _package;
    Property                        _qmpack_opt;
    
    // GWBSE options
    string                          _gwbse;
    Property                        _gwbse_opt;
    int                             _state;

    // XJob logbook (file output)
    string                          _outFile;
    //bool                            _energies2File;
    
    Property                       *_options;
};

// ========================================================================== //
//                      PARALLELCALC MEMBER FUNCTIONS                         //
// ========================================================================== //


void QMAPE::Initialize(Property *opt) {
    
	_options = opt;

    cout << endl
         << "... ... Initialized with " << _nThreads << " threads. "
         << flush;

    _maverick = (_nThreads == 1) ? true : false;

    string key = "options.qmape.jobcontrol";
		if ( opt->exists(key+".job_file")) {
			_jobfile = opt->get(key+".job_file").as<string>();
		}
		else {
			cout << endl;
			throw std::runtime_error("Job-file not set. Abort.");
		}

	key = "options.ewald.multipoles";
		if (opt->exists(key+".mapping")) {
			_xml_file = opt->get(key+".mapping").as< string >();
		}
		else {
			cout << endl;
			throw std::runtime_error("Multipole mapping file not set. Abort.");
		}
		if ( opt->exists(key+".mps_table")) {
			_mps_table = opt->get(key+".mps_table").as<string>();
		}
		else {
			cout << endl;
			throw std::runtime_error("Background mps table not set. Abort.");
		}
		if (opt->exists(key+".polar_bg")) {
			_polar_bg_arch = opt->get(key+".polar_bg").as<string>();
		}
		else { _polar_bg_arch = ""; }
		if (opt->exists(key+".pdb_check")) {
			_pdb_check = opt->get(key+".pdb_check").as<bool>();
		}
		else { _pdb_check = false; }
		if (opt->exists(key+".ptop_check")) {
			_ptop_check = opt->get(key+".ptop_check").as<bool>();
		}
		else { _ptop_check = false; }

    
    key = "options.qmape.qmpackage";
        if ( opt->exists(key+".package")) {
            string package_xml = opt->get(key+".package").as< string >();
            load_property_from_xml(_qmpack_opt, package_xml.c_str());
            _package = _qmpack_opt.get("package.name").as< string >();
        }
        else {
            throw runtime_error("No QM package specified.");
        }
    

    key = "options.qmape.gwbse";
    if ( opt->exists(key)) { 
    	cout << endl << "... ... Configure for excited states (DFT+GWBSE)" << flush;
        if ( opt->exists(key+".gwbse_options")) {
            string gwbse_xml = opt->get(key+".gwbse_options").as< string >();
            load_property_from_xml(_gwbse_opt, gwbse_xml.c_str());
            // _gwbse = _gwbse_opt.get("package.name").as< string >();
        }
        else {
            throw runtime_error("GWBSE options not specified.");
        }
        _state = opt->get(key+".state").as< int >();
    }
    else {
        cout << endl << "... ... Configure for ground states (DFT)" << flush;
    }

    QMPackageFactory::RegisterAll();
}


void QMAPE::PreProcess(Topology *top) {
    // INITIALIZE MPS-MAPPER (=> POLAR TOP PREP)
    cout << endl << "... ... Initialize MPS-mapper: " << flush;
    _mps_mapper.GenerateMap(_xml_file, _mps_table, top);
    return;
}


void QMAPE::CustomizeLogger(QMThread *thread) {
    
    // CONFIGURE LOGGER
    Logger* log = thread->getLogger();
    log->setReportLevel(logDEBUG);
    log->setMultithreading(_maverick);

    log->setPreface(logINFO,    (format("\nT%1$02d INF ...") % thread->getId()).str());
    log->setPreface(logERROR,   (format("\nT%1$02d ERR ...") % thread->getId()).str());
    log->setPreface(logWARNING, (format("\nT%1$02d WAR ...") % thread->getId()).str());
    log->setPreface(logDEBUG,   (format("\nT%1$02d DBG ...") % thread->getId()).str());        
}


// ========================================================================== //
//                            QMAPE MEMBER FUNCTIONS                          //
// ========================================================================== //


XJob QMAPE::ProcessInputString(Job *job, Topology *top, QMThread *thread) {

    // Input string looks like this:
    // <id1>:<name1>:<mpsfile1> <id2>:<name2>: ... ... ...

    string input = job->getInput().as<string>();
    vector<Segment*> qmSegs;
    vector<string>   qmSegMps;
    vector<string> split;
    Tokenizer toker(input, " \t\n");
    toker.ToVector(split);

    for (unsigned i = 0; i < split.size(); ++i) {

        string id_seg_mps = split[i];
        vector<string> split_id_seg_mps;
        Tokenizer toker(id_seg_mps, ":");
        toker.ToVector(split_id_seg_mps);

        int segId = boost::lexical_cast<int>(split_id_seg_mps[0]);
        string segName = split_id_seg_mps[1];
        string mpsFile = split_id_seg_mps[2];

        Segment *seg = top->getSegment(segId);
        if (seg->getName() != segName) {
            LOG(logERROR,*(thread->getLogger()))
                << "ERROR: Seg " << segId << ":" << seg->getName() << " "
                << " maltagged as " << segName << ". Skip job ..." << flush;
            throw std::runtime_error("Input does not match topology.");
        }

        qmSegs.push_back(seg);
        qmSegMps.push_back(mpsFile);
    }

    return XJob(job->getId(), job->getTag(), qmSegs, qmSegMps, top);
}


Job::JobResult QMAPE::EvalJob(Topology *top, Job *job, QMThread *thread) {
    
    // SILENT LOGGER FOR QMPACKAGE
    Logger* log = thread->getLogger();    
    Logger* qlog = new Logger();
    qlog->setReportLevel(logDEBUG);
    qlog->setMultithreading(_maverick);
    qlog->setPreface(logINFO,    (format("\nQ%1$02d ... ...") % thread->getId()).str());
    qlog->setPreface(logERROR,   (format("\nQ%1$02d ERR ...") % thread->getId()).str());
    qlog->setPreface(logWARNING, (format("\nQ%1$02d WAR ...") % thread->getId()).str());
    qlog->setPreface(logDEBUG,   (format("\nQ%1$02d DBG ...") % thread->getId()).str());

    // CREATE XJOB FROM JOB INPUT STRING
    LOG(logINFO,*log)
        << "Job input = " << job->getInput().as<string>() << flush;
    XJob xjob = this->ProcessInputString(job, top, thread);  

	// SETUP POLAR TOPOLOGY (GENERATE VS LOAD IF PREPOLARIZED)
	if (_polar_bg_arch == "") {
		LOG(logINFO,*log) << "Mps-Mapper: Generate FGC FGN BGN" << flush;
		_mps_mapper.Gen_FGC_FGN_BGN(top, &xjob, thread);
	}
	else {
		LOG(logINFO,*log) << "Mps-Mapper: Generate FGC, load FGN BGN from '"
				<< _polar_bg_arch << "'" << flush;
		_mps_mapper.Gen_FGC_Load_FGN_BGN(top, &xjob, _polar_bg_arch, thread);
	}
    LOG(logINFO,*log) << xjob.getPolarTop()->ShellInfoStr() << flush;

    // SETUP MM METHOD
    PEwald3D3D cape = PEwald3D3D(top, xjob.getPolarTop(), _options,
		thread->getLogger());
	if (_pdb_check)
		cape.WriteDensitiesPDB(xjob.getTag()+".densities.pdb");

    // SETUP QM HANDLERS
    QMPackage *qmpack =  QMPackages().Create(_package);
    qmpack->Initialize(&_qmpack_opt);
    qmpack->setLog(qlog);
    
    // SETUP QMAPE
    QMAPEMachine<QMPackage> machine = QMAPEMachine<QMPackage>(&xjob, &cape, qmpack,
        _options, "options.qmape", _subthreads);
    machine.setLog(thread->getLogger());
    
    // EVALUATE: ITERATE UNTIL CONVERGED
    machine.Evaluate(&xjob);

    // GENERATE OUTPUT AND FORWARD TO PROGRESS OBSERVER (RETURN)
    Job::JobResult jres = Job::JobResult();
    jres.setOutput(xjob.getInfoLine());
    jres.setStatus(Job::COMPLETE);
    
    // CLEAN-UP
    delete qmpack;
    delete qlog;

    return jres;
}



    
}}

#endif /* __QMAPE__H */
