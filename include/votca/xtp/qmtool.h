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


#ifndef __VOTCA_XTP_QMTOOL__H
#define __VOTCA_XTP_QMTOOL__H


#include <votca/tools/property.h>
#include <votca/tools/calculator.h>
#include <boost/format.hpp>


namespace votca { namespace xtp {
    
class QMTool : public votca::tools::Calculator
{
public:

    QMTool() { };
    virtual        ~QMTool() { };

    virtual std::string  Identify() = 0;
    virtual void    Initialize(votca::tools::Property *options) = 0;    
    virtual bool    Evaluate() = 0;
    virtual bool    EndEvaluate() { return true; }

protected:

};

}}

#endif /* _VOTCA_XTP_QMTOOL_H */
