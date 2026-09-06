from ackermann_adaptive_controller.flight_report import read_log, sessions


def test_quick_relaunch_is_a_new_session_without_changing_old_csv_schema(tmp_path):
    path=tmp_path/'flight.csv'
    path.write_text('stamp,phase,cmd_v\n100,RUN,0.2\n# run_id launch:one\n101,RUN,0.2\n102,RUN,0.3\n# run_id launch:two\n103,RUN,0.2\n')
    _,rows=read_log(path)
    assert [len(s) for s in sessions(rows)] == [1,2,1]


def test_legacy_session_gap_still_works(tmp_path):
    path=tmp_path/'flight.csv'; path.write_text('stamp,phase\n100,RUN\n101,RUN\n200,RUN\n')
    _,rows=read_log(path)
    assert [len(s) for s in sessions(rows)] == [2,1]
