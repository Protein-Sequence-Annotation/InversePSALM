import pytest

from inverse_psalm.inference.annotations import DomainSpan, build_annotation_track, parse_domain_spec


def test_parse_domain_spec_handles_pfam_and_none():
    assert parse_domain_spec("PF00042:25-175") == DomainSpan("PF00042", 25, 175)
    assert parse_domain_spec("None:176-250") == DomainSpan(None, 176, 250)


def test_build_annotation_track_uses_inclusive_coordinates():
    mapping = {"None": 0, "PF00001": {"start": 1, "middle": 2, "stop": 3}}
    track = build_annotation_track(
        length=5,
        domains=[DomainSpan("PF00001", 2, 4)],
        label_mapping=mapping,
        ignore_label=-100,
    )

    assert track.annotation_ids == [-100, -100, 1, 2, 3, -100, -100]
    assert track.residue_mask == [0, 1, 1, 1, 1, 1, 0]
    assert track.domain_mask == [0, 0, 1, 1, 1, 0, 0]


def test_build_annotation_track_allows_adjacent_but_rejects_overlap():
    mapping = {
        "None": 0,
        "PF00001": {"start": 1, "middle": 2, "stop": 3},
        "PF00002": {"start": 4, "middle": 5, "stop": 6},
    }
    track = build_annotation_track(
        length=10,
        domains=[DomainSpan("PF00001", 2, 4), DomainSpan("PF00002", 5, 7)],
        label_mapping=mapping,
    )
    assert track.annotation_ids[4] == 3
    assert track.annotation_ids[5] == 4

    with pytest.raises(ValueError, match="overlap"):
        build_annotation_track(
            length=10,
            domains=[DomainSpan("PF00001", 2, 4), DomainSpan("PF00002", 4, 7)],
            label_mapping=mapping,
        )


def test_explicit_none_span_injects_none_label():
    mapping = {"None": 0}
    track = build_annotation_track(
        length=3,
        domains=[DomainSpan(None, 2, 2)],
        label_mapping=mapping,
        ignore_label=-100,
    )

    assert track.annotation_ids == [-100, -100, 0, -100, -100]
    assert track.residue_mask == [0, 1, 1, 1, 0]
    assert track.domain_mask == [0, 0, 0, 0, 0]


def test_background_mode_none_uses_none_label_for_unspecified_positions():
    mapping = {"None": 0, "PF00001": {"start": 1, "middle": 2, "stop": 3}}
    track = build_annotation_track(
        length=3,
        domains=[DomainSpan("PF00001", 2, 2)],
        label_mapping=mapping,
        background_mode="none",
    )

    assert track.annotation_ids == [-100, 0, 1, 0, -100]
