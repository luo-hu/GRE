#include"./indexInterface.h"
#include "./alex/alex.h"
#include "./alexol/alex.h"
#include "./artsync/artrowex.h"
#include "./artsync/artolc.h"
#include "./artsync/artunsync.h"
#include "./xindex/xindex.h"
#include "./btreeolc/btreeolc.h"
#include "./hot/hot.h"
#include "./hot/hotrowex.h"
#include "./lipp/lipp.h"
#include "./lipphybrid/lipphybrid.h"
#include "./lippol/lippol.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
// #include "wormhole/wormhole.h"
#include "wormhole_u64/wormhole_u64.h"
#include "masstree/masstree.h"
#include "finedex/finedex.h"
#include "iostream"

template<class KEY_TYPE, class PAYLOAD_TYPE>
indexInterface<KEY_TYPE, PAYLOAD_TYPE> *get_index(std::string index_type) {
  indexInterface<KEY_TYPE, PAYLOAD_TYPE> *index;
  if (index_type == "alexol") {
    index = new alexolInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if(index_type == "alex") {
    index = new alexInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "btreeolc") {
    index = new BTreeOLCInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  // else if (index_type == "wormhole") {
  //   index = new WormholeInterface<KEY_TYPE, PAYLOAD_TYPE>;
  // }
  else if (index_type == "wormhole_u64") {
    index = new WormholeU64Interface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if( index_type == "hot") {
    index = new HotInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if( index_type == "hotrowex") {
    index = new HotRowexInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "masstree") {
    index = new MasstreeInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "xindex") {
    index = new xindexInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "pgm") {
    index = new pgmInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if(index_type == "btree") {
    index = new BTreeInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "artolc") {
    index = new ARTOLCInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  // else if (index_type == "artrowex") {
  //   index = new ARTROWEXInterface<KEY_TYPE, PAYLOAD_TYPE>;
  // }
  else if (index_type == "artunsync") {
    index = new ARTUnsynchronizedInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "lippol") {
    index = new LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "lipp") {
    index = new LIPPInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "lipphybrid") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "lipphybrid_identity") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, LIPPHybridKernelPolicy::IDENTITY_ONLY>;
  }
  else if (index_type == "lipphybrid_fallback") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, LIPPHybridKernelPolicy::FALLBACK_ONLY>;
  }
  else if (index_type == "lipphybrid_aggressive") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, LIPPHybridKernelPolicy::ADAPTIVE_AGGRESSIVE>;
  }
  else if (index_type == "lipphybrid_log1p") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, LIPPHybridKernelPolicy::FORCE_LOG1P>;
  }
  else if (index_type == "lipphybrid_sqrt") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, LIPPHybridKernelPolicy::FORCE_SQRT>;
  }
  else if (index_type == "lipphybrid_compact") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE,
                                    LIPPHybridKernelPolicy::IDENTITY_ONLY,
                                    LIPPHybridCompactPolicy::FORCE>;
  }
  else if (index_type == "lipphybrid_adaptive_compact") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE,
                                    LIPPHybridKernelPolicy::IDENTITY_ONLY,
                                    LIPPHybridCompactPolicy::ADAPTIVE>;
  }
  else if (index_type == "lipphybrid_aggressive_compact") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE,
                                    LIPPHybridKernelPolicy::ADAPTIVE_AGGRESSIVE,
                                    LIPPHybridCompactPolicy::FORCE>;
  }
  else if (index_type == "lipphybrid_aggressive_adaptive_compact") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE,
                                    LIPPHybridKernelPolicy::ADAPTIVE_AGGRESSIVE,
                                    LIPPHybridCompactPolicy::ADAPTIVE>;
  }
  else if (index_type == "lipphybrid_cbrt") {
    index = new LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, LIPPHybridKernelPolicy::FORCE_CBRT>;
  }
  else if (index_type == "finedex") {
    index = new finedexInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else {
    std::cout << "Could not find a matching index called " << index_type << ".\n";
    exit(0);
  }

  return index;
}
