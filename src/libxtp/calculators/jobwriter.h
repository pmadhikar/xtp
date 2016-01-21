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

#ifndef _VOTCA_XTP_JOBWRITER_H
#define _VOTCA_XTP_JOBWRITER_H

#include<votca/xtp/topology.h>
#include<votca/xtp/qmcalculator.h>


namespace votca { namespace xtp {
    
  
class JobWriter : public QMCalculator
{

public:

    typedef void (JobWriter::*WriteFunct)(Topology*);
    
    string Identify() { return "jobwriter"; }
    void Initialize(Property *options);
    bool EvaluateFrame(Topology *top);    
    
    // NEED TO REGISTER ALL WRITE MEMBERS IN ::Initialize
    void mps_ct(Topology *top);
    void mps_chrg(Topology *top);
    void mps_kmc(Topology *top);
    void mps_background(Topology *top);
    void mps_single(Topology *top);
    
    void edft(Topology *top);
    void idft(Topology *top);
    

private:

    Property *_options;
    vector<string> _keys;
    map<string,WriteFunct> _key_funct;
};




    
    
    
}}

#endif