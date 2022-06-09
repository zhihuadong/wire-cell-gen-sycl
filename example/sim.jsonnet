// Energy depo -> detector response simulation -> waveform frames

// local reality = std.extVar('reality');
local engine = std.extVar('engine');

local reality = 'data';
//local engine = 'Pgrapher';

local g = import 'pgraph.jsonnet';
local f = import 'pgrapher/common/funcs.jsonnet';
local wc = import 'wirecell.jsonnet';

local io = import 'pgrapher/common/fileio.jsonnet';
local tools_maker = import 'pgrapher/common/tools.jsonnet';

local data_params = import 'pgrapher/experiment/pdsp/params.jsonnet';
local simu_params = import 'pgrapher/experiment/pdsp/simparams.jsonnet';
local params = if reality == 'data' then data_params else simu_params;
local params = import 'pgrapher/experiment/pdsp/simparams.jsonnet';

// local tools = tools_maker(params);
local tools_orig = tools_maker(params);
local tools = tools_orig {
    anodes : [tools_orig.anodes[0],],
};

local sim_maker = import 'pgrapher/experiment/pdsp/sim.jsonnet';
local sim = sim_maker(params, tools);

local nanodes = std.length(tools.anodes);
local anode_iota = std.range(0, nanodes-1);

local wcls_maker = import "pgrapher/ui/wcls/nodes.jsonnet";
local wcls = wcls_maker(params, tools);
local wcls_input = {
    depos: wcls.input.depos(name="", art_tag="IonAndScint"),
};

// Collect all the wc/ls output converters for use below.  Note the
// "name" MUST match what is used in theh "outputers" parameter in the
// FHiCL that loads this file.
local mega_anode = {
  type: 'MegaAnodePlane',
  name: 'meganodes',
  data: {
    anodes_tn: [wc.tn(anode) for anode in tools.anodes],
  },
};
local wcls_output = {
    // ADC output from simulation
    // sim_digits: wcls.output.digits(name="simdigits", tags=["orig"]),
    sim_digits: g.pnode({
      type: 'wclsFrameSaver',
      name: 'simdigits',
      data: {
        // anode: wc.tn(tools.anode),
        anode: wc.tn(mega_anode),
        digitize: true,  // true means save as RawDigit, else recob::Wire
        frame_tags: ['daq'],
        // nticks: params.daq.nticks,
        // chanmaskmaps: ['bad'],
      },
    }, nin=1, nout=1, uses=[mega_anode]),

    // The noise filtered "ADC" values.  These are truncated for
    // art::Event but left as floats for the WCT SP.  Note, the tag
    // "raw" is somewhat historical as the output is not equivalent to
    // "raw data".
    nf_digits: wcls.output.digits(name="nfdigits", tags=["raw"]),

    // The output of signal processing.  Note, there are two signal
    // sets each created with its own filter.  The "gauss" one is best
    // for charge reconstruction, the "wiener" is best for S/N
    // separation.  Both are used in downstream WC code.
    sp_signals: wcls.output.signals(name="spsignals", tags=["gauss", "wiener"]),

    // save "threshold" from normal decon for each channel noise
    // used in imaging
    sp_thresholds: wcls.output.thresholds(name="spthresholds", tags=["threshold"]),
};

//local deposio = io.numpy.depos(output);
local drifter = sim.drifter;
local bagger = [sim.make_bagger("bagger%d"%n) for n in anode_iota];

// signal plus noise pipelines
local sn_pipes = sim.signal_pipelines;
//local sn_pipes = sim.splusn_pipelines;

// local perfect = import 'pgrapher/experiment/pdsp/chndb-perfect.jsonnet';
local base = import 'pgrapher/experiment/pdsp/chndb-base.jsonnet';
local chndb = [{
  type: 'OmniChannelNoiseDB',
  name: 'ocndbperfect%d' % n,
  // data: perfect(params, tools.anodes[n], tools.field, n),
  data: base(params, tools.anodes[n], tools.field, n),
  uses: [tools.anodes[n], tools.field],  // pnode extension
} for n in anode_iota];

local nf_maker = import 'pgrapher/experiment/pdsp/nf.jsonnet';
local nf_pipes = [nf_maker(params, tools.anodes[n], chndb[n], n, name='nf%d' % n) for n in anode_iota];

local sp_maker = import 'pgrapher/experiment/pdsp/sp.jsonnet';
local sp = sp_maker(params, tools, { sparse: true, use_roi_debug_mode: false, use_multi_plane_protection: false });
local sp_pipes = [sp.make_sigproc(a) for a in tools.anodes];

local deposplats = [sim.make_ductor('splat%d'%n, tools.anodes[n], tools.pirs[0], 'DepoSplat') for n in anode_iota] ;

local magoutput = 'g4-rec-0.root';
local magnify = import 'pgrapher/experiment/pdsp/magnify-sinks.jsonnet';
local sinks = magnify(tools, magoutput);

local hio_truth = [g.pnode({
      type: 'HDF5FrameTap',
      name: 'hio_truth%d' % n,
      data: {
        anode: wc.tn(tools.anodes[n]),
        trace_tags: ['ductor%d'%n],
        filename: "g4-tru-%d.h5" % n,
        chunk: [0, 0], // ncol, nrow
        gzip: 2,
        high_throughput: false,
      },  
    }, nin=1, nout=1),
    for n in std.range(0, std.length(tools.anodes) - 1)
    ];

local kokkos_test = [g.pnode({
      type: 'KokkosTestFrameFilter',
      name: 'kokkos_test%d' % n,
      data: {
      },  
    }, nin=1, nout=1),
    for n in std.range(0, std.length(tools.anodes) - 1)
    ];

local hio_orig = [g.pnode({
      type: 'HDF5FrameTap',
      name: 'hio_orig%d' % n,
      data: {
        anode: wc.tn(tools.anodes[n]),
        trace_tags: ['orig%d'%n],
        filename: "g4-rec-%d.h5" % n,
        chunk: [0, 0], // ncol, nrow
        gzip: 2,
        high_throughput: false,
      },  
    }, nin=1, nout=1),
    for n in std.range(0, std.length(tools.anodes) - 1)
    ];

local hio_sp = [g.pnode({
      type: 'HDF5FrameTap',
      name: 'hio_sp%d' % n,
      data: {
        anode: wc.tn(tools.anodes[n]),
        trace_tags: ['gauss%d' % n,
        ],
        filename: "g4-rec-%d.h5" % n,
        chunk: [0, 0], // ncol, nrow
        gzip: 2,
        high_throughput: false,
      },  
    }, nin=1, nout=1),
    for n in std.range(0, std.length(tools.anodes) - 1)
    ];

local retagger = [g.pnode({
        type: 'Retagger',
        data: {
            // Note: retagger keeps tag_rules an array to be like frame fanin/fanout.
            tag_rules: [{
            // Retagger also handles "frame" and "trace" like fanin/fanout
            // merge separately all traces like gaussN to gauss.
            frame: {
                '.*': 'orig%d'%n,
            },
            }],
        },
        }, nin=1, nout=1),
    for n in std.range(0, std.length(tools.anodes) - 1)
    ];


local reco_fork = [
    g.pipeline([
                bagger[n],
                sn_pipes[n],
                retagger[n],
                // kokkos_test[n],
                hio_orig[n],
                // nf_pipes[n],
                // sp_pipes[n],
                // hio_sp[n],
                g.pnode({ type: 'DumpFrames', name: 'reco_fork%d'%n }, nin=1, nout=0),
                ],
                'reco_fork%d' % n)
    for n in anode_iota
];


local depo_fanout = [g.pnode({
    type:'DepoFanout',
    name:'depo_fanout-%d'%n,
    data:{
        multiplicity:2,
        tags: [],
    }}, nin=1, nout=2) for n in anode_iota];
local frame_fanin = [g.pnode({
    type: 'FrameFanin',
    name: 'frame_fanin-%d'%n,
    data: {
        multiplicity: 2, 
        tags: [],
    }}, nin=2, nout=1) for n in anode_iota];

local frame_sink = g.pnode({ type: 'DumpFrames' }, nin=1, nout=0);

local multipass = [reco_fork[n] for n in anode_iota];

local outtags = ['orig%d' % n for n in anode_iota];


local depo_fanout_1st = g.pnode({
    type:'DepoFanout',
    name:'depo_fanout_1st',
    data:{
        multiplicity:nanodes,
        tags: [],
    }}, nin=1, nout=nanodes);

local retagger = g.pnode({
  type: 'Retagger',
  data: {
    // Note: retagger keeps tag_rules an array to be like frame fanin/fanout.
    tag_rules: [{
      // Retagger also handles "frame" and "trace" like fanin/fanout
      // merge separately all traces like gaussN to gauss.
      frame: {
        '.*': 'orig',
      },
      merge: {
        'orig\\d': 'daq',
      },
    }],
  },
}, nin=1, nout=1);

//local frameio = io.numpy.frames(output);
local sink = sim.frame_sink;

// g4 sim as input
local graph = g.intern(innodes=[wcls_input.depos], centernodes=[drifter, depo_fanout_1st]+multipass, outnodes=[],
                      edges = 
                      [
                        g.edge(wcls_input.depos, drifter, 0, 0),
                        g.edge(drifter, depo_fanout_1st, 0, 0),
                      ] +
                      [g.edge(depo_fanout_1st, multipass[n],  n, 0) for n in anode_iota],
                      );

local app = {
  type: engine,
  data: {
    edges: g.edges(graph),
  },
};

local env = {
    type: "SyclEnv",
};


// Finally, the configuration sequence which is emitted.

[env] + g.uses(graph) + [app]
