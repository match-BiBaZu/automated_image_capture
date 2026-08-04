"""Read-only BLE diagnostic: list advertisements and optionally GATT services."""

from __future__ import annotations

import argparse
import asyncio

from bleak import BleakClient, BleakScanner


async def main(address: str | None, timeout: float, address_type: str | None) -> None:
    devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
    print(f"Visible BLE devices: {len(devices)}")
    for device, advertisement in devices.values():
        manufacturer_data = {
            hex(key): value.hex()
            for key, value in advertisement.manufacturer_data.items()
        }
        service_data = {
            key: value.hex() for key, value in advertisement.service_data.items()
        }
        raw_advertisement = getattr(device.details, "adv", None)
        advertisement_type = getattr(raw_advertisement, "advertisement_type", None)
        bluetooth_address_type = getattr(
            raw_advertisement, "bluetooth_address_type", None
        )
        print(
            f"{device.name or '<unnamed>'} {device.address} "
            f"RSSI={advertisement.rssi}\n"
            f"  local_name={advertisement.local_name!r}\n"
            f"  services={advertisement.service_uuids}\n"
            f"  manufacturer_data={manufacturer_data}\n"
            f"  service_data={service_data}\n"
            f"  advertisement_type={advertisement_type}"
            f" address_type={bluetooth_address_type}"
        )
    if address:
        target = next(
            (
                device
                for device, _advertisement in devices.values()
                if device.address.casefold() == address.casefold()
            ),
            address,
        )
        winrt = {"address_type": address_type} if address_type else {}
        async with BleakClient(target, timeout=8.0, winrt=winrt) as client:
            print(f"GATT services for {address}:")
            for service in client.services:
                print(f"  {service.uuid} {service.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", metavar="ADDRESS", help="Read GATT services from one device")
    parser.add_argument("--address-type", choices=("public", "random"))
    parser.add_argument("--timeout", type=float, default=8.0, help="Passive scan time in seconds")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.connect, arguments.timeout, arguments.address_type))
