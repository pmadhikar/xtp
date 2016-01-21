/*
 *            Copyright 2009-2016 The VOTCA Development Team
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


#ifndef __VOTCA_MD2QM_StateSaverSQLite_H
#define	__VOTCA_MD2QM_StateSaverSQLite_H

#include <stdio.h>
#include <map>
#include "qmdatabase.h"
#include "topology.h"
#include <boost/interprocess/sync/file_lock.hpp>

namespace votca { namespace xtp {

using namespace votca::tools;

class StateSaverSQLite
{
public:
    StateSaverSQLite() { };
   ~StateSaverSQLite() { _db.Close(); }

    void Open(Topology &qmtop, const string &file, bool lock = true);
    void Close() { _db.Close(); }
    bool NextFrame();

    void WriteFrame();
    void WriteMeta(bool update);
    void WriteMolecules(bool update);
    void WriteSegTypes(bool update);
    void WriteSegments(bool update);
    void WriteFragments(bool update);
    void WriteAtoms(bool update);
    void WritePairs(bool update);
    void WriteSuperExchange(bool update);

    void ReadFrame();
    void ReadMeta(int topId);
    void ReadMolecules(int topId);
    void ReadSegTypes(int topId);
    void ReadSegments(int topId);
    void ReadFragments(int topId);
    void ReadAtoms(int topId);
    void ReadPairs(int topId);
    void ReadSuperExchange(int topId);

    int  FramesInDatabase();
    Topology *getTopology() { return _qmtop; }
    bool HasTopology(Topology *top);
    
    void LockStateFile();
    void UnlockStateFile();
    
private:
    Topology       *_qmtop;
    QMDatabase      _db;

    int             _frame;
    int             _current_frame;
    vector<int>     _frames;
    vector<int>     _topIds;

    string          _sqlfile;
    bool            _was_read;
    
    boost::interprocess::file_lock *_flock;
};

}}

#endif	/* __VOTCA_MD2QM_StateSaverSQLite2_H */

