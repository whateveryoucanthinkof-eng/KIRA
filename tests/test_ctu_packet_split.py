"""CTU-13 binetflow has no per-direction packet count: split it by the known
byte direction, never 50/50 (which gave one-way flows invented replies)."""

from data_unification.ctu13_adapter import CTU13Adapter

HDR = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"


def _parse(tmp_path, rows):
    p = tmp_path / "x.binetflow"
    p.write_text(HDR + "".join(rows))
    return list(CTU13Adapter().parse_netflow_csv(str(p)))


def test_one_way_flow_has_no_reply_packets(tmp_path):
    rows = ["2011/08/10 09:46:53.047277,1.0,tcp,1.1.1.1,1025,->,2.2.2.2,80,S_,0,0,10,600,600,"
            "flow=From-Botnet-V42-TCP-Attempt\n"]
    (r,) = _parse(tmp_path, rows)
    assert (r.fwd_packets, r.bwd_packets) == (10, 0)


def test_two_way_flow_splits_by_bytes(tmp_path):
    rows = ["2011/08/10 09:46:53.047277,1.0,tcp,1.1.1.1,1025,<->,2.2.2.2,80,FSPA_FSPA,0,0,10,10000,1000,"
            "flow=Background-TCP-Established\n"]
    (r,) = _parse(tmp_path, rows)
    assert r.fwd_packets + r.bwd_packets == 10
    assert r.fwd_packets >= 1 and r.bwd_packets > r.fwd_packets
