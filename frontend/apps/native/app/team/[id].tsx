import React from "react";
import {
  View,
  Text,
  ScrollView,
  FlatList,
  TouchableOpacity,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Loading } from "@/components/Loading";
import { Player } from "@nba/api/types";

export default function TeamDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const teamId = parseInt(id, 10);
  const router = useRouter();

  const { data: teamRoster, isLoading, error } = useQuery({
    queryKey: ["team", teamId, "roster"],
    queryFn: () => apiClient.getTeamRoster(teamId),
    enabled: !!teamId,
  });

  const handlePlayerPress = (player: Player) => {
    router.push(`/player/${player.person_id}`);
  };

  if (isLoading) {
    return (
      <View className="flex-1 bg-gray-50">
        <Loading text="Loading team..." />
      </View>
    );
  }

  if (error || !teamRoster) {
    return (
      <View className="flex-1 bg-gray-50 items-center justify-center">
        <Text className="text-red-500 text-lg">Team not found</Text>
      </View>
    );
  }

  return (
    <ScrollView className="flex-1 bg-gray-50">
      {/* Team Header */}
      <View className="bg-white p-6 shadow-sm">
        <Text className="text-3xl font-bold text-gray-900 mb-2">
          {teamRoster.team.full_name}
        </Text>
        <View className="flex-row items-center">
          <View className="bg-blue-100 px-3 py-1 rounded-full">
            <Text className="text-blue-800 font-bold">
              {teamRoster.team.team_abbrev}
            </Text>
          </View>
        </View>
      </View>

      {/* Roster */}
      <View className="bg-white mx-4 my-4 rounded-lg shadow-sm p-4">
        <Text className="text-lg font-bold text-gray-900 mb-4">Roster</Text>
        {teamRoster.roster.length === 0 ? (
          <Text className="text-gray-500 text-center py-4">
            No roster data available
          </Text>
        ) : (
          <FlatList
            data={teamRoster.roster}
            keyExtractor={(item) => item.person_id.toString()}
            renderItem={({ item }) => (
              <TouchableOpacity
                onPress={() => handlePlayerPress(item)}
                className="py-3 border-b border-gray-100 last:border-b-0"
              >
                <View className="flex-row justify-between items-center">
                  <View className="flex-1">
                    <Text className="font-semibold text-gray-900">
                      {item.full_name}
                    </Text>
                    <Text className="text-gray-600 mt-1">
                      {item.primary_position}
                    </Text>
                  </View>
                  {item.height && item.body_weight && (
                    <Text className="text-gray-500 text-sm">
                      {item.height}" • {item.body_weight} lbs
                    </Text>
                  )}
                </View>
              </TouchableOpacity>
            )}
            scrollEnabled={false}
            showsVerticalScrollIndicator={false}
          />
        )}
      </View>
    </ScrollView>
  );
}