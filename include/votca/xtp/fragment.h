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

#ifndef __VOTCA_XTP_FRAGMENT_H
#define	__VOTCA_XTP_FRAGMENT_H

#include <votca/xtp/atom.h>
#include <votca/xtp/polarsite.h>
#include <votca/xtp/apolarsite.h>
#include <fstream>



namespace votca { namespace xtp {

class Topology;
class Molecule;
class Segment;
    
/**
    \brief Rigid fragment. One conjugated segment contains several rigid fragments.

 * Apart from the position and orientation it has a pointer to a conjugated segment 
 * it belongs to as well as positions of atoms which belong to it of two types those 
 * which are generated by MD and those obtained by substitution of rigid copies
 */

class Fragment {
public:

     Fragment(int id, string name) : _id(id), _name(name), _symmetry(-1) { }
     Fragment(Fragment *stencil);
    ~Fragment();
    
    void Rotate(matrix spin, vec refPos);    // rotates w.r.t. center of map
    void TranslateBy(const vec &shift);
    void RotTransQM2MD();

    inline void setTopology(Topology *container) { _top = container; }
    inline void setMolecule(Molecule *container) { _mol = container; }
    inline void setSegment(Segment *container)   { _seg = container; }
    void        AddAtom( Atom* atom );
    void        AddPolarSite(PolarSite *pole);
    void        AddAPolarSite(APolarSite *pole);

    Topology            *getTopology() { return _top; }
    Molecule            *getMolecule() { return _mol; }
    Segment             *getSegment()  { return _seg; }
    vector< Atom* >     &Atoms() { return _atoms; }
    vector<PolarSite*>  &PolarSites() { return _polarSites; }
    vector<APolarSite*> &APolarSites() { return _apolarSites; }

    const int    &getId() const { return _id; }
    const string &getName() const { return _name; }

    void         Rigidify(bool Auto = 0);
    void         setSymmetry(int sym) { _symmetry = sym; }
    const int   &getSymmetry() { return _symmetry; }
    void         setTrihedron(vector<int> trihedron) { _trihedron = trihedron; }
    const vector< int > &getTrihedron() { return _trihedron; }


    void          calcPos(string tag = "MD");
    void          setPos(vec pos) { _CoMD = pos; }
    const vec    &getPos() const { return _CoMD; }
    const vec    &getCoMD() { return _CoMD; }
    const vec    &getCoQM() { return _CoQM; }
    const vec    &getCoQM0() { return _CoQM0; }
    const matrix &getRotQM2MD() { return _rotateQM2MD; }
    const vec    &getTransQM2MD() { return _translateQM2MD; }
    
    
    

private:

    Segment     *_seg;

    vector < Atom* > _atoms;
    vector <PolarSite*> _polarSites;
    vector <APolarSite*> _apolarSites;
    vector< double > _weights;

    int         _id;
    string      _name;
    Topology    *_top;
    Molecule    *_mol;
    int              _symmetry;

    matrix      _rotateQM2MD;       // Set via ::Rigidify()
    vec         _CoQM;              // Center of map (QM)
    vec         _CoMD;              // Center of map (MD)
    vector< int >    _trihedron;
    vec         _CoQM0;             // Center of map (QM) original (for IZindo)
    vec         _translateQM2MD;    // Set via ::Rigidify()


};

}}

#endif	/* __VOTCA_XTP_FRAGMENT_H */

