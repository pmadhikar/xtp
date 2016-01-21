#include <votca/xtp/parallelxjobcalc.h>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

using boost::format;


namespace votca { namespace xtp {

    
template<typename JobContainer, typename pJob, typename rJob> 
bool ParallelXJobCalc<JobContainer,pJob,rJob>::EvaluateFrame(Topology *top) {    
   
    // RIGIDIFY TOPOLOGY (=> LOCAL FRAMES)
    if (!top->isRigid()) {
        bool isRigid = top->Rigidify();
        if (!isRigid) { return 0; }
    }
    else cout << endl << "... ... System is already rigidified." << flush;
    
    // CONVERT THREADS INTO SUBTHREADS IF BENEFICIAL
    if (_XJobs.size() < _nThreads && false) {
        _subthreads = (_nThreads - _XJobs.size()) / _XJobs.size() + 1;
        _nThreads   = _XJobs.size();

        cout << endl << "... ... "
             << "Converted threads into subthreads to increase efficiency: "
             << "NT = " << _nThreads << ", NST = " << _subthreads
             << flush;
    }

    // INITIALIZE PROGRESS OBSERVER
    string progFile = _jobfile;
    assert(_jobfile != "__NOFILE__");    
    JobOperator* master = new JobOperator(-1, top, this);    
    master->getLogger()->setReportLevel(logDEBUG);
    master->getLogger()->setMultithreading(true);
    master->getLogger()->setPreface(logINFO,    "\nMST INF");
    master->getLogger()->setPreface(logERROR,   "\nMST ERR");
    master->getLogger()->setPreface(logWARNING, "\nMST WAR");
    master->getLogger()->setPreface(logDEBUG,   "\nMST DBG");    
    _progObs->InitFromProgFile(progFile, master);

    // PRE-PROCESS (OVERWRITTEN IN CHILD OBJECT)
    this->PreProcess(top);
    
    // CREATE + EXECUTE THREADS (XJOB HANDLERS)
    vector<JobOperator*> jobOps;

    for (unsigned int id = 0; id < _nThreads; id++) {
        JobOperator *newOp = new JobOperator(id, top, this);
        jobOps.push_back(newOp);
    }
    
    for (unsigned int id = 0; id < _nThreads; ++id) {
        CustomizeLogger(jobOps[id]);
    }

    for (unsigned int id = 0; id < _nThreads; id++) {
        jobOps[id]->InitData(top);
    }

    if (!_maverick) cout << endl; // REQUIRED FOR PROGRESS BAR IN OBSERVER
    
    for (unsigned int id = 0; id < _nThreads; id++) {
        jobOps[id]->Start();
    }

    for (unsigned int id = 0; id < _nThreads; id++) {
        jobOps[id]->WaitDone();
    }
    
    if (!_maverick)
    for (unsigned int id = 0; id < _nThreads; id++) {
        cout << endl << *(jobOps[id]->getLogger()) << flush;
    }

    for (unsigned int id = 0; id < _nThreads; id++) {
        delete jobOps[id];
    }    

    jobOps.clear();

	// SYNC REMAINING COMPLETE JOBS
	_progObs->SyncWithProgFile(master);
    
    // POST-PROCESS (OVERWRITTEN IN CHILD OBJECT)
    this->PostProcess(top);
    
    return true;
}


template<typename JobContainer, typename pJob, typename rJob>
void ParallelXJobCalc<JobContainer,pJob,rJob>::JobOperator::Run(void) {

    while (true) {
        _job = _master->_progObs->RequestNextJob(this);

        if (_job == NULL) { break; }
        else { 
            rJob res = this->_master->EvalJob(_top, _job, this);
            this->_master->_progObs->ReportJobDone(_job, &res, this);
        }
    }
}

template<typename JobContainer, typename pJob, typename rJob>
void ParallelXJobCalc<JobContainer,pJob,rJob>::CustomizeLogger(QMThread *thread) {
    
    // CONFIGURE LOGGER
    Logger* log = thread->getLogger();
    log->setReportLevel(logDEBUG);
    log->setMultithreading(_maverick);

    log->setPreface(logINFO,    (format("\nT%1$02d INF ...") % thread->getId()).str());
    log->setPreface(logERROR,   (format("\nT%1$02d ERR ...") % thread->getId()).str());
    log->setPreface(logWARNING, (format("\nT%1$02d WAR ...") % thread->getId()).str());
    log->setPreface(logDEBUG,   (format("\nT%1$02d DBG ...") % thread->getId()).str());        
}

// REGISTER PARALLEL CALCULATORS
//template class ParallelXJobCalc< vector<XJob*>, XJob* >;
//template class ParallelXJobCalc< vector<Segment*>, Segment* >;
//template class ParallelXJobCalc< QMNBList, QMPair* >;
template class ParallelXJobCalc< vector<Job*>, Job*, Job::JobResult >;

}}
