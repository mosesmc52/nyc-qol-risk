import re
from typing import Dict, Optional

LEVEL_1_CATEGORIES = [
    "Housing & Buildings",
    "Noise & Disturbance",
    "Sanitation & Waste",
    "Streets & Infrastructure",
    "Water & Sewer",
    "Public Safety & Behavior",
    "Transportation & Vehicles",
    "Health & Environment",
    "Parks & Trees",
    "Animals & Pests",
    "Consumer & Business",
    "Homelessness & Social Services",
]

LEVEL_2_BY_LEVEL_1 = {
    "Housing & Buildings": [
        "Heat & Hot Water",
        "Plumbing & Water Systems",
        "Structural Issues",
        "Indoor Hazards",
        "Building Systems",
    ],
    "Noise & Disturbance": [
        "Residential Noise",
        "Commercial Noise",
        "Construction Noise",
        "Traffic & Vehicle Noise",
        "Aircraft Noise",
        "General Noise",
    ],
    "Sanitation & Waste": [
        "Garbage Collection",
        "Recycling",
        "Illegal Dumping",
        "Unsanitary Conditions",
        "Street Cleanliness",
    ],
    "Streets & Infrastructure": [
        "Roadway Conditions",
        "Sidewalk & Curb Issues",
        "Street Lighting",
        "Traffic Signals",
        "Signage",
        "Public Fixtures",
        "Bridge & Tunnel Conditions",
    ],
    "Water & Sewer": [
        "Water Leaks",
        "Sewer Issues",
        "Flooding & Drainage",
        "Water Quality",
        "Water System Maintenance",
    ],
    "Public Safety & Behavior": [
        "Substance Use",
        "Disorderly Conduct",
        "Public Nuisance",
        "Non-Emergency Police Matters",
        "Crowds & Gatherings",
        "Safety Hazards",
    ],
    "Transportation & Vehicles": [
        "Parking Issues",
        "Blocked Access",
        "Abandoned Vehicles",
        "Taxi & For-Hire Vehicles",
        "Bus Stops & Transit Fixtures",
        "Micromobility",
        "Traffic Conditions",
    ],
    "Health & Environment": [
        "Air Quality",
        "Food Safety",
        "Hazardous Materials",
        "Public Health Violations",
        "Environmental Monitoring",
    ],
    "Parks & Trees": [
        "Tree Maintenance",
        "Tree Requests",
        "Park Maintenance",
        "Park Rule Violations",
        "Recreation Facilities",
    ],
    "Animals & Pests": [
        "Rodents",
        "Insects",
        "Domestic Animals",
        "Animal Welfare",
        "Wildlife & Other Animal Issues",
    ],
    "Consumer & Business": [
        "Retail Complaints",
        "Vendor Enforcement",
        "Food Establishments",
        "Regulated Businesses",
        "Service Complaints",
    ],
    "Homelessness & Social Services": [
        "Homeless Assistance",
        "Encampments",
        "Service Delivery Issues",
    ],
}


def _clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    v = value.strip().lower()
    v = v.replace("_", " ")
    v = re.sub(r"\s+", " ", v)
    v = re.sub(r"[^a-z0-9\s/&-]", "", v)
    return v


def normalize_category(category: Optional[str]) -> Dict[str, str]:
    """
    Normalize a raw NYC 311 complaint type into:
      - normalized_input
      - level_1
      - level_2

    Returns a dict like:
    {
        "raw": "Noise - Residential",
        "normalized_input": "noise - residential",
        "level_1": "Noise & Disturbance",
        "level_2": "Residential Noise",
    }
    """
    raw = category or ""
    c = _clean_text(category)

    if not c:
        return {
            "raw": raw,
            "normalized_input": c,
            "level_1": "Public Safety & Behavior",
            "level_2": "Public Nuisance",
        }

    # ------------------------------------------------------------
    # Exact / near-exact mappings for high-confidence routing
    # ------------------------------------------------------------
    exact_map = {
        # Housing & Buildings
        "heat/hot water": ("Housing & Buildings", "Heat & Hot Water"),
        "non-residential heat": ("Housing & Buildings", "Heat & Hot Water"),
        "plumbing": ("Housing & Buildings", "Plumbing & Water Systems"),
        "general construction/plumbing": (
            "Housing & Buildings",
            "Plumbing & Water Systems",
        ),
        "boiler": ("Housing & Buildings", "Heat & Hot Water"),
        "boilers": ("Housing & Buildings", "Heat & Hot Water"),
        "paint/plaster": ("Housing & Buildings", "Structural Issues"),
        "mold": ("Housing & Buildings", "Indoor Hazards"),
        "lead": ("Housing & Buildings", "Indoor Hazards"),
        "asbestos": ("Housing & Buildings", "Indoor Hazards"),
        "elevator": ("Housing & Buildings", "Building Systems"),
        "door/window": ("Housing & Buildings", "Building Systems"),
        "flooring/stairs": ("Housing & Buildings", "Building Systems"),
        "building condition": ("Housing & Buildings", "Structural Issues"),
        "building/use": ("Housing & Buildings", "Structural Issues"),
        "facades": ("Housing & Buildings", "Structural Issues"),
        "facade insp safety pgm": ("Housing & Buildings", "Structural Issues"),
        "unstable building": ("Housing & Buildings", "Structural Issues"),
        "building drinking water tank": (
            "Housing & Buildings",
            "Plumbing & Water Systems",
        ),
        # Noise
        "noise": ("Noise & Disturbance", "General Noise"),
        "noise - residential": ("Noise & Disturbance", "Residential Noise"),
        "noise - commercial": ("Noise & Disturbance", "Commercial Noise"),
        "noise - vehicle": ("Noise & Disturbance", "Traffic & Vehicle Noise"),
        "noise - helicopter": ("Noise & Disturbance", "Aircraft Noise"),
        "noise - house of worship": ("Noise & Disturbance", "General Noise"),
        "noise - park": ("Noise & Disturbance", "General Noise"),
        "noise - street/sidewalk": ("Noise & Disturbance", "Traffic & Vehicle Noise"),
        # Sanitation & Waste
        "missed collection": ("Sanitation & Waste", "Garbage Collection"),
        "request large bulky item collection": (
            "Sanitation & Waste",
            "Garbage Collection",
        ),
        "illegal dumping": ("Sanitation & Waste", "Illegal Dumping"),
        "dumpster complaint": ("Sanitation & Waste", "Garbage Collection"),
        "unsanitary condition": ("Sanitation & Waste", "Unsanitary Conditions"),
        "dirty condition": ("Sanitation & Waste", "Street Cleanliness"),
        "litter basket complaint": ("Sanitation & Waste", "Street Cleanliness"),
        "litter basket request": ("Sanitation & Waste", "Street Cleanliness"),
        "recycling basket complaint": ("Sanitation & Waste", "Recycling"),
        "residential disposal complaint": ("Sanitation & Waste", "Garbage Collection"),
        "commercial disposal complaint": ("Sanitation & Waste", "Garbage Collection"),
        "institution disposal complaint": ("Sanitation & Waste", "Garbage Collection"),
        "electronics waste appointment": ("Sanitation & Waste", "Recycling"),
        "dead animal": ("Sanitation & Waste", "Unsanitary Conditions"),
        # Streets & Infrastructure
        "street condition": ("Streets & Infrastructure", "Roadway Conditions"),
        "highway condition": ("Streets & Infrastructure", "Roadway Conditions"),
        "dep highway condition": ("Streets & Infrastructure", "Roadway Conditions"),
        "sidewalk condition": ("Streets & Infrastructure", "Sidewalk & Curb Issues"),
        "dep sidewalk condition": (
            "Streets & Infrastructure",
            "Sidewalk & Curb Issues",
        ),
        "curb condition": ("Streets & Infrastructure", "Sidewalk & Curb Issues"),
        "street light condition": ("Streets & Infrastructure", "Street Lighting"),
        "traffic signal condition": ("Streets & Infrastructure", "Traffic Signals"),
        "street sign - damaged": ("Streets & Infrastructure", "Signage"),
        "street sign - dangling": ("Streets & Infrastructure", "Signage"),
        "street sign - missing": ("Streets & Infrastructure", "Signage"),
        "highway sign - damaged": ("Streets & Infrastructure", "Signage"),
        "highway sign - dangling": ("Streets & Infrastructure", "Signage"),
        "highway sign - missing": ("Streets & Infrastructure", "Signage"),
        "bridge condition": ("Streets & Infrastructure", "Bridge & Tunnel Conditions"),
        "tunnel condition": ("Streets & Infrastructure", "Bridge & Tunnel Conditions"),
        "broken parking meter": ("Streets & Infrastructure", "Public Fixtures"),
        "public payphone complaint": ("Streets & Infrastructure", "Public Fixtures"),
        "linknyc": ("Streets & Infrastructure", "Public Fixtures"),
        "wayfinding": ("Streets & Infrastructure", "Public Fixtures"),
        "bench": ("Streets & Infrastructure", "Public Fixtures"),
        # Water & Sewer
        "water leak": ("Water & Sewer", "Water Leaks"),
        "water quality": ("Water & Sewer", "Water Quality"),
        "water conservation": ("Water & Sewer", "Water System Maintenance"),
        "water system": ("Water & Sewer", "Water System Maintenance"),
        "water maintenance": ("Water & Sewer", "Water System Maintenance"),
        "sewer": ("Water & Sewer", "Sewer Issues"),
        "sewer maintenance": ("Water & Sewer", "Sewer Issues"),
        "indoor sewage": ("Water & Sewer", "Sewer Issues"),
        "standing water": ("Water & Sewer", "Flooding & Drainage"),
        "water drainage": ("Water & Sewer", "Flooding & Drainage"),
        "root/sewer/sidewalk condition": ("Water & Sewer", "Sewer Issues"),
        # Public Safety & Behavior
        "drug activity": ("Public Safety & Behavior", "Substance Use"),
        "drinking": ("Public Safety & Behavior", "Substance Use"),
        "panhandling": ("Public Safety & Behavior", "Public Nuisance"),
        "squeegee": ("Public Safety & Behavior", "Public Nuisance"),
        "disorderly youth": ("Public Safety & Behavior", "Disorderly Conduct"),
        "non-emergency police matter": (
            "Public Safety & Behavior",
            "Non-Emergency Police Matters",
        ),
        "mass gathering complaint": ("Public Safety & Behavior", "Crowds & Gatherings"),
        "illegal fireworks": ("Public Safety & Behavior", "Safety Hazards"),
        "scaffold safety": ("Public Safety & Behavior", "Safety Hazards"),
        "safety": ("Public Safety & Behavior", "Safety Hazards"),
        # Transportation & Vehicles
        "illegal parking": ("Transportation & Vehicles", "Parking Issues"),
        "blocked driveway": ("Transportation & Vehicles", "Blocked Access"),
        "abandoned vehicle": ("Transportation & Vehicles", "Abandoned Vehicles"),
        "derelict vehicles": ("Transportation & Vehicles", "Abandoned Vehicles"),
        "taxi complaint": ("Transportation & Vehicles", "Taxi & For-Hire Vehicles"),
        "taxi compliment": ("Transportation & Vehicles", "Taxi & For-Hire Vehicles"),
        "taxi report": ("Transportation & Vehicles", "Taxi & For-Hire Vehicles"),
        "taxi licensee complaint": (
            "Transportation & Vehicles",
            "Taxi & For-Hire Vehicles",
        ),
        "green taxi complaint": (
            "Transportation & Vehicles",
            "Taxi & For-Hire Vehicles",
        ),
        "dispatched taxi complaint": (
            "Transportation & Vehicles",
            "Taxi & For-Hire Vehicles",
        ),
        "dispatched taxi compliment": (
            "Transportation & Vehicles",
            "Taxi & For-Hire Vehicles",
        ),
        "for hire vehicle complaint": (
            "Transportation & Vehicles",
            "Taxi & For-Hire Vehicles",
        ),
        "for hire vehicle report": (
            "Transportation & Vehicles",
            "Taxi & For-Hire Vehicles",
        ),
        "fhv licensee complaint": (
            "Transportation & Vehicles",
            "Taxi & For-Hire Vehicles",
        ),
        "bus stop shelter complaint": (
            "Transportation & Vehicles",
            "Bus Stops & Transit Fixtures",
        ),
        "bus stop shelter placement": (
            "Transportation & Vehicles",
            "Bus Stops & Transit Fixtures",
        ),
        "bike rack": ("Transportation & Vehicles", "Micromobility"),
        "bike rack condition": ("Transportation & Vehicles", "Micromobility"),
        "bike/roller/skate": ("Transportation & Vehicles", "Micromobility"),
        "bike/roller/skate chronic": ("Transportation & Vehicles", "Micromobility"),
        "e-scooter": ("Transportation & Vehicles", "Micromobility"),
        "traffic": ("Transportation & Vehicles", "Traffic Conditions"),
        "municipal parking facility": ("Transportation & Vehicles", "Parking Issues"),
        # Health & Environment
        "air quality": ("Health & Environment", "Air Quality"),
        "indoor air quality": ("Health & Environment", "Air Quality"),
        "food establishment": ("Health & Environment", "Food Safety"),
        "food poisoning": ("Health & Environment", "Food Safety"),
        "hazardous materials": ("Health & Environment", "Hazardous Materials"),
        "radioactive material": ("Health & Environment", "Hazardous Materials"),
        "x-ray machine/equipment": ("Health & Environment", "Hazardous Materials"),
        "cooling tower": ("Health & Environment", "Public Health Violations"),
        "smoking": ("Health & Environment", "Public Health Violations"),
        "smoking or vaping": ("Health & Environment", "Public Health Violations"),
        "tanning": ("Health & Environment", "Public Health Violations"),
        "tattooing": ("Health & Environment", "Public Health Violations"),
        "trans fat": ("Health & Environment", "Public Health Violations"),
        "calorie labeling": ("Health & Environment", "Public Health Violations"),
        "bottled water": ("Health & Environment", "Public Health Violations"),
        # Parks & Trees
        "damaged tree": ("Parks & Trees", "Tree Maintenance"),
        "dead/dying tree": ("Parks & Trees", "Tree Maintenance"),
        "overgrown tree/branches": ("Parks & Trees", "Tree Maintenance"),
        "illegal tree damage": ("Parks & Trees", "Tree Maintenance"),
        "uprooted stump": ("Parks & Trees", "Tree Maintenance"),
        "new tree request": ("Parks & Trees", "Tree Requests"),
        "plant": ("Parks & Trees", "Park Maintenance"),
        "violation of park rules": ("Parks & Trees", "Park Rule Violations"),
        "animal in a park": ("Parks & Trees", "Park Maintenance"),
        "beach/pool/sauna complaint": ("Parks & Trees", "Recreation Facilities"),
        "lifeguard": ("Parks & Trees", "Recreation Facilities"),
        "public toilet": ("Parks & Trees", "Recreation Facilities"),
        # Animals & Pests
        "rodent": ("Animals & Pests", "Rodents"),
        "mosquitoes": ("Animals & Pests", "Insects"),
        "harboring bees/wasps": ("Animals & Pests", "Insects"),
        "unleashed dog": ("Animals & Pests", "Domestic Animals"),
        "unlicensed dog": ("Animals & Pests", "Domestic Animals"),
        "animal-abuse": ("Animals & Pests", "Animal Welfare"),
        "illegal animal kept as pet": ("Animals & Pests", "Animal Welfare"),
        "illegal animal sold": ("Animals & Pests", "Animal Welfare"),
        "pet sale": ("Animals & Pests", "Animal Welfare"),
        "pet shop": ("Animals & Pests", "Animal Welfare"),
        "animal facility - no permit": ("Animals & Pests", "Animal Welfare"),
        "unsanitary animal facility": ("Animals & Pests", "Animal Welfare"),
        "unsanitary animal pvt property": ("Animals & Pests", "Animal Welfare"),
        "unsanitary pigeon condition": (
            "Animals & Pests",
            "Wildlife & Other Animal Issues",
        ),
        # Consumer & Business
        "consumer complaint": ("Consumer & Business", "Retail Complaints"),
        "retailer complaint": ("Consumer & Business", "Retail Complaints"),
        "vendor enforcement": ("Consumer & Business", "Vendor Enforcement"),
        "mobile food vendor": ("Consumer & Business", "Vendor Enforcement"),
        "cannabis retailer": ("Consumer & Business", "Regulated Businesses"),
        "outdoor dining": ("Consumer & Business", "Regulated Businesses"),
        "transfer station complaint": ("Consumer & Business", "Service Complaints"),
        # Homelessness & Social Services
        "homeless person assistance": (
            "Homelessness & Social Services",
            "Homeless Assistance",
        ),
        "encampment": ("Homelessness & Social Services", "Encampments"),
        "home delivered meal - missed delivery": (
            "Homelessness & Social Services",
            "Service Delivery Issues",
        ),
    }

    if c in exact_map:
        level_1, level_2 = exact_map[c]
        return {
            "raw": raw,
            "normalized_input": c,
            "level_1": level_1,
            "level_2": level_2,
        }

    # ------------------------------------------------------------
    # Rule-based routing for categories not in exact_map
    # ------------------------------------------------------------

    # Housing & Buildings
    if any(k in c for k in ["heat", "hot water", "boiler"]):
        return _result(raw, c, "Housing & Buildings", "Heat & Hot Water")

    if any(k in c for k in ["plumbing", "building drinking water tank"]):
        return _result(raw, c, "Housing & Buildings", "Plumbing & Water Systems")

    if any(
        k in c
        for k in [
            "building condition",
            "building/use",
            "facade",
            "unstable building",
            "posted notice",
            "construction safety enforcement",
            "stalled sites",
        ]
    ):
        return _result(raw, c, "Housing & Buildings", "Structural Issues")

    if any(k in c for k in ["mold", "lead", "asbestos"]):
        return _result(raw, c, "Housing & Buildings", "Indoor Hazards")

    if any(
        k in c
        for k in [
            "elevator",
            "door/window",
            "window guard",
            "flooring/stairs",
            "appliance",
            "electric",
            "electrical",
        ]
    ):
        return _result(raw, c, "Housing & Buildings", "Building Systems")

    # Noise
    if "noise" in c:
        if "residential" in c:
            return _result(raw, c, "Noise & Disturbance", "Residential Noise")
        if "commercial" in c:
            return _result(raw, c, "Noise & Disturbance", "Commercial Noise")
        if any(k in c for k in ["construction", "site safety", "cranes", "derricks"]):
            return _result(raw, c, "Noise & Disturbance", "Construction Noise")
        if any(k in c for k in ["vehicle", "street", "sidewalk"]):
            return _result(raw, c, "Noise & Disturbance", "Traffic & Vehicle Noise")
        if "helicopter" in c or "aircraft" in c:
            return _result(raw, c, "Noise & Disturbance", "Aircraft Noise")
        return _result(raw, c, "Noise & Disturbance", "General Noise")

    # Sanitation & Waste
    if any(
        k in c
        for k in [
            "collection",
            "bulky item",
            "dumpster",
            "disposal",
            "sanitation worker",
        ]
    ):
        return _result(raw, c, "Sanitation & Waste", "Garbage Collection")

    if "recycling" in c or "electronics waste" in c:
        return _result(raw, c, "Sanitation & Waste", "Recycling")

    if "dumping" in c:
        return _result(raw, c, "Sanitation & Waste", "Illegal Dumping")

    if any(k in c for k in ["unsanitary", "dirty", "dead animal"]):
        return _result(raw, c, "Sanitation & Waste", "Unsanitary Conditions")

    if any(k in c for k in ["litter", "street sweeping"]):
        return _result(raw, c, "Sanitation & Waste", "Street Cleanliness")

    # Streets & Infrastructure
    if any(
        k in c
        for k in ["street condition", "highway condition", "dep street condition"]
    ):
        return _result(raw, c, "Streets & Infrastructure", "Roadway Conditions")

    if any(k in c for k in ["sidewalk", "curb"]):
        return _result(raw, c, "Streets & Infrastructure", "Sidewalk & Curb Issues")

    if "street light" in c:
        return _result(raw, c, "Streets & Infrastructure", "Street Lighting")

    if "traffic signal" in c:
        return _result(raw, c, "Streets & Infrastructure", "Traffic Signals")

    if "sign" in c:
        return _result(raw, c, "Streets & Infrastructure", "Signage")

    if any(
        k in c for k in ["bench", "payphone", "parking meter", "linknyc", "wayfinding"]
    ):
        return _result(raw, c, "Streets & Infrastructure", "Public Fixtures")

    if any(k in c for k in ["bridge", "tunnel"]):
        return _result(raw, c, "Streets & Infrastructure", "Bridge & Tunnel Conditions")

    # Water & Sewer
    if "water leak" in c:
        return _result(raw, c, "Water & Sewer", "Water Leaks")

    if any(k in c for k in ["sewer", "indoor sewage"]):
        return _result(raw, c, "Water & Sewer", "Sewer Issues")

    if any(k in c for k in ["drainage", "standing water", "flood", "drain"]):
        return _result(raw, c, "Water & Sewer", "Flooding & Drainage")

    if "water quality" in c or "drinking water" in c:
        return _result(raw, c, "Water & Sewer", "Water Quality")

    if "water" in c:
        return _result(raw, c, "Water & Sewer", "Water System Maintenance")

    # Public Safety & Behavior
    if any(k in c for k in ["drug", "drinking", "urinating"]):
        return _result(raw, c, "Public Safety & Behavior", "Substance Use")

    if any(k in c for k in ["disorderly"]):
        return _result(raw, c, "Public Safety & Behavior", "Disorderly Conduct")

    if any(k in c for k in ["panhandling", "squeegee", "quality of life"]):
        return _result(raw, c, "Public Safety & Behavior", "Public Nuisance")

    if "police" in c:
        return _result(
            raw, c, "Public Safety & Behavior", "Non-Emergency Police Matters"
        )

    if any(k in c for k in ["mass gathering", "crowd"]):
        return _result(raw, c, "Public Safety & Behavior", "Crowds & Gatherings")

    if any(k in c for k in ["fireworks", "safety", "obstruction"]):
        return _result(raw, c, "Public Safety & Behavior", "Safety Hazards")

    # Transportation & Vehicles
    if any(k in c for k in ["parking", "parking meter"]):
        return _result(raw, c, "Transportation & Vehicles", "Parking Issues")

    if "blocked driveway" in c:
        return _result(raw, c, "Transportation & Vehicles", "Blocked Access")

    if any(k in c for k in ["abandoned vehicle", "derelict vehicle"]):
        return _result(raw, c, "Transportation & Vehicles", "Abandoned Vehicles")

    if any(k in c for k in ["taxi", "for hire vehicle", "fhv"]):
        return _result(raw, c, "Transportation & Vehicles", "Taxi & For-Hire Vehicles")

    if any(k in c for k in ["bus stop shelter", "ferry"]):
        return _result(
            raw, c, "Transportation & Vehicles", "Bus Stops & Transit Fixtures"
        )

    if any(k in c for k in ["bike", "skate", "roller", "e-scooter"]):
        return _result(raw, c, "Transportation & Vehicles", "Micromobility")

    if "traffic" in c:
        return _result(raw, c, "Transportation & Vehicles", "Traffic Conditions")

    # Health & Environment
    if "air quality" in c:
        return _result(raw, c, "Health & Environment", "Air Quality")

    if any(k in c for k in ["food", "poisoning"]):
        return _result(raw, c, "Health & Environment", "Food Safety")

    if any(
        k in c
        for k in [
            "hazardous",
            "radioactive",
            "x-ray",
            "oil or gas spill",
            "industrial waste",
        ]
    ):
        return _result(raw, c, "Health & Environment", "Hazardous Materials")

    if any(
        k in c
        for k in [
            "smoking",
            "vaping",
            "cooling tower",
            "trans fat",
            "calorie",
            "tanning",
            "tattooing",
            "vaccine mandate",
        ]
    ):
        return _result(raw, c, "Health & Environment", "Public Health Violations")

    if any(k in c for k in ["sustainability", "water conservation"]):
        return _result(raw, c, "Health & Environment", "Environmental Monitoring")

    # Parks & Trees
    if any(k in c for k in ["tree", "stump"]):
        if "request" in c:
            return _result(raw, c, "Parks & Trees", "Tree Requests")
        return _result(raw, c, "Parks & Trees", "Tree Maintenance")

    if any(k in c for k in ["park", "plant", "special natural area district", "snad"]):
        return _result(raw, c, "Parks & Trees", "Park Maintenance")

    if "violation of park rules" in c:
        return _result(raw, c, "Parks & Trees", "Park Rule Violations")

    if any(k in c for k in ["beach", "pool", "lifeguard", "sauna", "public toilet"]):
        return _result(raw, c, "Parks & Trees", "Recreation Facilities")

    # Animals & Pests
    if "rodent" in c:
        return _result(raw, c, "Animals & Pests", "Rodents")

    if any(k in c for k in ["mosquito", "bees", "wasps", "poison ivy"]):
        return _result(raw, c, "Animals & Pests", "Insects")

    if any(k in c for k in ["dog", "domestic"]):
        return _result(raw, c, "Animals & Pests", "Domestic Animals")

    if any(
        k in c
        for k in [
            "animal-abuse",
            "animal facility",
            "illegal animal",
            "pet shop",
            "pet sale",
        ]
    ):
        return _result(raw, c, "Animals & Pests", "Animal Welfare")

    if "animal" in c or "pigeon" in c:
        return _result(raw, c, "Animals & Pests", "Wildlife & Other Animal Issues")

    # Consumer & Business
    if any(k in c for k in ["consumer complaint", "retailer complaint"]):
        return _result(raw, c, "Consumer & Business", "Retail Complaints")

    if any(k in c for k in ["vendor", "mobile food vendor"]):
        return _result(raw, c, "Consumer & Business", "Vendor Enforcement")

    if "food establishment" in c:
        return _result(raw, c, "Consumer & Business", "Food Establishments")

    if any(
        k in c
        for k in [
            "cannabis retailer",
            "outdoor dining",
            "private school vaccine mandate",
        ]
    ):
        return _result(raw, c, "Consumer & Business", "Regulated Businesses")

    if any(
        k in c
        for k in [
            "complaint",
            "compliment",
            "report",
            "found property",
            "lost property",
        ]
    ):
        return _result(raw, c, "Consumer & Business", "Service Complaints")

    # Homelessness & Social Services
    if "homeless" in c:
        return _result(raw, c, "Homelessness & Social Services", "Homeless Assistance")

    if "encampment" in c:
        return _result(raw, c, "Homelessness & Social Services", "Encampments")

    if any(k in c for k in ["meal", "service", "delivery"]):
        return _result(
            raw, c, "Homelessness & Social Services", "Service Delivery Issues"
        )

    # ------------------------------------------------------------
    # Final fallback
    # ------------------------------------------------------------
    return {
        "raw": raw,
        "normalized_input": c,
        "level_1": "Public Safety & Behavior",
        "level_2": "Public Nuisance",
    }


def _result(
    raw: str, normalized_input: str, level_1: str, level_2: str
) -> Dict[str, str]:
    return {
        "raw": raw,
        "normalized_input": normalized_input,
        "level_1": level_1,
        "level_2": level_2,
    }
