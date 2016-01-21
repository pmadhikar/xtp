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


#ifndef _QMPair_H
#define _QMPair_H

#include "segment.h"
#include <utility>


namespace votca { namespace xtp {

class Topology;



class QMPair : public std::pair< Segment*, Segment* >
{
public:
    
    enum PairType 
    { 
        Hopping,
        SuperExchange,
        SuperExchangeAndHopping,
        Excitoncl,
        
    };

    QMPair() :  _R(0,0,0),
                _ghost(NULL), 
                _top(NULL),
                _id(-1),   
                _hasGhost(0),
                _rate12_e(0),
                _rate21_e(0),
                _rate12_h(0),
                _rate21_h(0),
                _has_e(false),
                _has_h(false),
                _lambdaO_e(0),
                _lambdaO_h(0),
                _Jeff2_e(0),
                _Jeff2_h(0),
                _rate12_s(0),
                _rate21_s(0),
                _rate12_t(0),
                _rate21_t(0),
                _has_s(false),
                _has_t(false),
                _lambdaO_s(0),
                _lambdaO_t(0),   
                _Jeff2_s(0),
                _Jeff2_t(0),
                _pair_type(Hopping) { };
    QMPair(int id, Segment *seg1, Segment *seg2);
   ~QMPair();


   int       getId() { return _id; }
   Topology *getTopology() { return _top; }
   void      setTopology(Topology *top) { _top = top; }
   vec      &R() { return _R; }
   double    Dist() { return abs(_R); }
   vec       getPos() { return 0.5*(first->getPos() + second->getPos()); }

   void     setIsPathCarrier(bool yesno, int carrier);
   bool     isPathCarrier(int carrier);

   void     setLambdaO(double lO, int carrier);
   double   getLambdaO(int carrier);
   
   double   getReorg12(int state) { return first->getU_nC_nN(state) + second->getU_cN_cC(state); } // 1->2
   double   getReorg21(int state) { return first->getU_cN_cC(state) + second->getU_nC_nN(state); } // 2->1
  
   double   getReorg12_x(int state) { return first->getU_nX_nN(state) + second->getU_xN_xX(state); } // 1->2
   double   getReorg21_x(int state) { return first->getU_xN_xX(state) + second->getU_nX_nN(state); } // 1->2

   void     setRate12(double rate, int state);
   void     setRate21(double rate, int state);
   double   getRate12(int state);
   double   getRate21(int state);
   vec      getR();

   void     setJs(const vector <double> Js, int state);
   double   calcJeff2(int state);
   double   getJeff2(int state) ;
   void     setJeff2(double Jeff2, int state);
   vector<double> &Js(int state);

   double   getdE12(int state) { return second->getSiteEnergy(state)
                                       -first->getSiteEnergy(state); }

   Segment* Seg1PbCopy() { return first; }
   Segment* Seg2PbCopy();
   Segment* Seg1() { return first; }
   Segment* Seg2() { return second; }

   bool     HasGhost() { return _hasGhost; }
   void     WritePDB(string fileName);
   void     WriteXYZ(FILE *out, bool useQMPos = true);

   // superexchange pairs have a list of bridging segments
   void     setType( PairType pair_type ) { _pair_type = pair_type; }
   void     setType( int pair_type ) { _pair_type = (PairType) pair_type; }
   void     AddBridgingSegment( Segment* _segment ){ _bridging_segments.push_back(_segment); }
   const vector<Segment*> &getBridgingSegments() const { return _bridging_segments; }
   PairType &getType(){return _pair_type;}

protected:

    vec         _R;

    Segment    *_ghost;
    Topology   *_top;
    int         _id;
    bool        _hasGhost;
    
    double _rate12_e;    // from ::Rates        output    DEFAULT 0
    double _rate21_e;    // from ::Rates        output    DEFAULT 0
    double _rate12_h;
    double _rate21_h;
    double _has_e;       // from ::Rates        input     DEFAULT 0
    double _has_h;
    double _lambdaO_e;   // from ::EOutersphere output    DEFAULT 0
    double _lambdaO_h;
    
    vector <double> _Js_e;
    vector <double> _Js_h;
    double          _Jeff2_e;
    double          _Jeff2_h;
    //excition part s:singlet t:triplet
    // state +2: singlet
    //state +3:triplet
    
    
    double _rate12_s;   
    double _rate21_s; 
    double _rate12_t;
    double _rate21_t;
    double _has_s;       
    double _has_t;
    double _lambdaO_s;   
    double _lambdaO_t;
    
    vector <double> _Js_s;
    vector <double> _Js_t;
    double          _Jeff2_s;
    double          _Jeff2_t;

    PairType _pair_type;
    vector<Segment*> _bridging_segments;


};

}}


#endif
