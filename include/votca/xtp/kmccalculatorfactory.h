/*
 * Copyright 2009-2016 The VOTCA Development Team (http://www.votca.org)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef __VOTCA_KMC_CALCULATORFACTORY_H
#define	__VOTCA_KMC_CALCULATORFACTORY_H

#include <votca/tools/objectfactory.h>
#include "kmccalculator.h"

namespace votca { namespace xtp {

using namespace std;
using namespace tools;

class KMCCalculatorFactory
: public ObjectFactory<std::string, KMCCalculator>
{
private:
    KMCCalculatorFactory() {}
public:
    
    static void RegisterAll(void);

    friend KMCCalculatorFactory &Calculators();
};

inline KMCCalculatorFactory &Calculators()
{
    static KMCCalculatorFactory _instance;
    return _instance;
}

}}

#endif	/* __VOTCA_KMC_CALCULATORFACTORY_H */

