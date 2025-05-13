"""MRIData datasets."""

import os
import ismrmrd
import ismrmrd.xsd
import numpy as np
from .base import BaseMRIDataset, RawMRI
from ..utils.network import get_data_path

MRIDATA_DOWNLOAD_URL = "http://mridata.org/download"
STANFORD_2D_FSE_UUIDs = [
    "b738ca00-bdf1-463c-901e-28ae7001c290",
    "346bd30b-36cc-4130-a335-bd294333189b",
    "5a9d6228-d1fd-4c36-83bb-a9eef50a7a87",
    "2e8ff5d1-b74c-4860-a5f3-c031104fc4bf",
    "53293c62-70d3-4834-9db9-572a69c49410",
    "1525b67d-f473-4ba8-b88c-f9c54e6e8475",
    "600f24e0-ae05-4b5a-9227-9df42720c738",
    "6f6fa401-2fe9-4cbb-9790-b70f927fb2ed",
    "91d23736-6bf6-405e-a555-eb7e7b086349",
    "dc65aadd-b474-48ef-a5e2-c9e7e0c4b5eb",
    "02a390fc-1836-45a1-9991-863a66d7536f",
    "5695b91b-201c-419e-8052-6020d07a9715",
    "c2623b8f-7b9a-4058-9587-6f62e82452d1",
    "345bc0eb-c3a7-4507-8718-53e9e5db3e85",
    "1e9890f0-7c2b-49a1-a835-bdf463e2cd5d",
    "20103d2f-9abf-4962-8bf1-f4acf54caff8",
    "f286f257-b311-4bed-8f11-816d29603002",
    "b61494ea-5013-4b2d-b204-12a19a89ef51",
    "bddedfa8-eb63-4eca-967c-2abce1c2b0d6",
    "efac30a2-8b30-48e8-9a7c-aad2a05a46df",
    "1edf6cd2-3027-4b4e-abfd-2a973037162b",
    "6b0f4577-10d1-4f8a-b321-3818a27dbbea",
    "846c5c2d-85a1-4f6b-895d-700f31c60e20",
    "292aa3a9-a287-4883-a272-b2ba2f7ca834",
    "d83306bd-b2a1-4263-8683-04777b63685c",
    "7cdb5257-d2b6-48e7-b29d-88572489f87a",
    "00321123-d1e5-4038-929e-0f230ddb33b7",
    "40861cf7-6646-4392-b8a5-0f180ebdb31e",
    "15ecb733-7067-4cf9-896c-38bb2096940e",
    "eaac7417-5a9c-4ab5-bc01-a732461ac8a5",
    "b4a9bfe6-c2e5-4e8f-ad30-e95807a55156",
    "63a31a04-76f1-41f5-b5bb-4a84ebecc31c",
    "3f66edb6-06c2-4a90-984b-50264cdce95f",
    "a70827b2-cf53-4c39-a67a-9540dc36a501",
    "f0f6bfaa-8f3d-401d-a62c-c0dffa86ccdc",
    "518e71db-5042-41df-9d46-16038ce594aa",
    "206f3e4c-e1d0-469a-a96b-3da1899fbb10",
    "94d77ec7-abfb-496b-a64b-5552d2e0fcf8",
    "fdc08f61-f844-409a-8a20-b44a2e6f2901",
    "f22f1d3a-6122-4f02-a8c9-54801651437f",
    "9fb36d02-2ae2-4a9e-b70e-614f6f73ba07",
    "f013be2c-3287-4e2c-b3d2-62cc07cb4f70",
    "3dc8a1a2-5d1e-4200-b04c-98f8b4b62a72",
    "0e0b0437-866c-4f0c-ac52-1e7592c159c6",
    "d31ba0e2-9af0-4014-802b-85158de159a2",
    "77db8c36-58a3-43e4-a15d-7d2069e7d725",
    "41ea5f8e-c2fb-4f5a-8fde-36bd7a66a2f1",
    "5eb20e5b-10af-4ec5-910a-a22f5424a60c",
    "34075fe3-532d-4168-ba85-3b0c050c71c7",
    "27abf29c-56b1-4c57-8265-96e65e47fa00",
    "70224135-71f5-47f7-bd32-6019930e9a15",
    "f0e12329-bf14-402c-a437-11ab0ad5ad6c",
    "f9cad7d2-e449-49b0-b82f-cd5e1d520df5",
    "6b12c40f-c03b-4343-8147-d500426f5db1",
    "1f8e6baf-d2f5-4a81-8f0f-fbec620550eb",
    "74318ad3-0d0a-4168-9f05-c094efc0ddd9",
    "dbc43d6f-f4dc-4df0-807a-1753cb8780da",
    "2d74779c-09c4-4dab-b7e3-83ab7e35be7f",
    "be13d3e7-aa50-459f-bb23-dd25a89130c8",
    "23f32ce4-1be8-41fb-baae-c990a2cd9a6d",
    "ee9a3fff-6bf1-46cc-86f4-9a2962c046ce",
    "6579d109-d4a0-45e4-907a-538ee801fbfb",
    "b5c89356-d7aa-4656-a306-6a5f3efa867c",
    "dd932c2d-0deb-40ea-9254-7e11055cac59",
    "638abe71-aa47-4eeb-bb50-f70fa9db8445",
    "8fed546a-3e3a-471a-8078-0c6a55a1f8aa",
    "bd42d696-9b54-4a68-af6e-7523b84429a8",
    "1c50a912-bc81-4578-9dfb-700738f5c3aa",
    "3f157697-59aa-4d50-95c1-7f3f6d7ec7fc",
    "5a2b25b3-df06-4cbb-8e09-811fca90ef3b",
    "daefd846-d57f-4e78-bd49-b2a29efd2623",
    "56215408-e5a5-426e-8222-a7d03caed9a9",
    "33a2987b-d868-4ba7-afd9-8a4534c3d515",
    "8db4a687-1f9b-4e4c-9078-305875368709",
    "e1f270cc-26ff-48cf-8e6a-f1b36ab71b28",
    "272ac55d-0c3b-4f10-a68f-841a452484a0",
    "7b2c6a8a-0cff-4eb1-84ed-7dd490563181",
    "f63bc805-31ae-416b-ad60-29fdef5be0a6",
    "48843afc-117d-4dfe-9cc9-9f09b9e062b1",
    "67524e41-9061-4a1a-a9a4-16ad2115e91b",
    "eb9a9a8f-7500-401a-b895-16461d6e7e7a",
    "1d1ee050-adc9-4b0c-9bbb-ce1c6ee044ee",
    "c265af27-f79a-4333-8643-1a0b4220d04c",
    "956bc107-490d-47e1-bb16-886418ac5962",
    "7720d579-7114-4ee8-a28b-0792ff06b0a6",
    "52455c3b-fd2d-4b8a-b24d-e1bd0d37345f",
    "5ee5b565-a26a-4f48-bf6c-36cc88d7773f",
    "2538bfed-e353-4dd8-bc95-0869f49fd79e",
    "4b00a5ea-2bb8-448b-b9e7-ba497bc3fb7d",
]


class StanfordKnee2D(BaseMRIDataset):
    def __init__(self, local_path=None):
        super().__init__("mridata", local_path=local_path)

    def __len__(self):
        return len(STANFORD_2D_FSE_UUIDs)

    def _data_path(self, idx, local_path=None, force_update=False, proxies=None):
        target_url = "{}/{}".format(MRIDATA_DOWNLOAD_URL, STANFORD_2D_FSE_UUIDs[idx])
        dest = get_data_path(
            target_url,
            self.uid,
            path=local_path,
            force_update=force_update,
            proxies=proxies,
        )
        return dest

    def _get_single_subject_data(self, idx):
        dest = self._data_path(idx, local_path=self.local_path)
        raw = RawMRI(dest)
        return raw
